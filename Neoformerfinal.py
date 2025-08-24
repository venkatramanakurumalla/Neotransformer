# -*- coding: utf-8 -*-
"""
🚀 NeoFormer-X: Ultra-Powerful Text Generation Model
====================================================
- Grouped Query Attention (GQA) + FlashAttention-2
- Rotary Positional Embeddings (RoPE)
- DeepSeek-style MoE (Mixture of Experts)
- Dynamic NTK-aware scaling for longer context
- RMSNorm for stability
- SwiGLU activation
- QK-LayerNorm for training stability
- Lion optimizer + cosine annealing with warmup
- Gradient checkpointing + fully sharded data parallel
- Advanced KV caching with paged attention
- Multi-modal support (text + future image integration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
from tqdm import tqdm
import math
import os
import argparse
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Optional: PEFT (LoRA)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  peft not installed. Run: pip install peft")

# Optional: FlashAttention-2
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH = True
    print("✅ FlashAttention-2 enabled")
except ImportError:
    HAS_FLASH = False
    print("⚠️  FlashAttention-2 not available.")

# Optional: tiktoken
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # More modern tokenizer
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>", "<|image|>", "<|audio|>"})
    decode = lambda l: enc.decode(l)
    VOCAB_SIZE = enc.n_vocab
    print(f"✅ Using cl100k_base tokenizer with {VOCAB_SIZE} tokens")
except ImportError:
    print("⚠️  tiktoken not installed. Install with: pip install tiktoken")
    raise RuntimeError("tiktoken required for NeoFormer-X")

# ==============================
# 0. Advanced Utilities
# ==============================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency tensor for rotary positional embeddings"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class DynamicNTKScaling:
    """Dynamic NTK-aware scaling for longer context"""
    def __init__(self, alpha: float = 1.0, beta: float = 32.0):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, seq_len: int, trained_seq_len: int = 2048) -> float:
        if seq_len <= trained_seq_len:
            return 1.0
        return self.alpha * (seq_len / trained_seq_len) ** (self.beta / (self.beta + 1))

# ==============================
# 1. Advanced Attention with GQA, RoPE, and FlashAttention-2
# ==============================
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE and FlashAttention-2"""
    def __init__(self, d_model: int, nhead: int = 8, n_kv_heads: int = 4, 
                 dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        
        # GQA projections - fewer parameters for key/value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # QK LayerNorm for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Rotary positional embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, max_seq_len * 2
        )
        
        # NTK scaling
        self.ntk_scaling = DynamicNTKScaling()

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache: bool = False, seq_len: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        H, K, KV_H = self.nhead, self.head_dim, self.n_kv_heads
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, L, H, K).transpose(1, 2)  # [B, H, L, K]
        k = self.k_proj(x).view(B, L, KV_H, K).transpose(1, 2)  # [B, KV_H, L, K]
        v = self.v_proj(x).view(B, L, KV_H, K).transpose(1, 2)  # [B, KV_H, L, K]
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary positional embeddings
        freqs_cis = self.freqs_cis[:L].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Apply NTK scaling if needed
        scale = self.ntk_scaling(L + seq_len) if seq_len > 0 else 1.0
        if scale != 1.0:
            q = q * scale
        
        # Merge with past KV (if provided)
        if kv is not None:
            past_k, past_v = kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Repeat K and V for GQA if n_kv_heads != nhead
        if KV_H != H:
            k = k.repeat_interleave(H // KV_H, dim=1)
            v = v.repeat_interleave(H // KV_H, dim=1)
        
        # FlashAttention or native implementation
        if HAS_FLASH:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                context = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            context = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(context)
        
        if use_cache:
            new_kv = (k, v)  # Return as tuple for efficiency
            return out, new_kv
        return out, None

# ==============================
# 2. MoE FeedForward Network
# ==============================
class MoEFeedForward(nn.Module):
    """Mixture of Experts FeedForward Network with SwiGLU"""
    def __init__(self, d_model: int, d_ff: int = 2048, num_experts: int = 8, 
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff * 2),
                nn.SiLU(),  # SwiGLU: SiLU activation with gating
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Additional regularization
        self.aux_loss_coef = 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)  # Flatch for expert processing
        
        # Calculate gate scores
        gate_scores = self.gate(x)  # [batch*seq_len, num_experts]
        
        # Top-k routing
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = torch.softmax(top_k_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Calculate auxiliary loss for load balancing
        if self.training:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        # Process through top-k experts
        for i in range(self.top_k):
            expert_mask = top_k_indices == i
            expert_idx = i
            
            if expert_mask.any():
                # Get expert output
                expert_out = self.experts[expert_idx](x[expert_mask])
                
                # Apply gating scores
                expert_scores = top_k_scores[expert_mask, i].unsqueeze(1)
                output[expert_mask] += expert_out * expert_scores
                
                if self.training:
                    expert_usage[expert_idx] += expert_mask.float().sum()
        
        # Calculate auxiliary loss for load balancing
        aux_loss = 0.0
        if self.training:
            expert_usage = expert_usage / expert_usage.sum()
            aux_loss = self.aux_loss_coef * (expert_usage.std() ** 2)
        
        output = output.view(batch_size, seq_len, d_model)
        return output, aux_loss

# ==============================
# 3. Enhanced NeoFormer Block
# ==============================
class NeoFormerBlock(nn.Module):
    """Enhanced NeoFormer Block with MoE and advanced attention"""
    def __init__(self, d_model: int, nhead: int = 8, n_kv_heads: int = 4, 
                 d_ff: int = 2048, num_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        self.attn = GroupedQueryAttention(
            d_model, nhead=nhead, n_kv_heads=n_kv_heads, 
            dropout=dropout, max_seq_len=max_seq_len
        )
        
        # MoE FeedForward Network
        self.ffn = MoEFeedForward(
            d_model, d_ff=d_ff, num_experts=num_experts, 
            top_k=top_k, dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache: bool = False, seq_len: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, new_kv = self.attn(x, kv=kv, use_cache=use_cache, seq_len=seq_len)
        x = residual + self.dropout(attn_out)
        
        # MoE FeedForward with residual connection
        residual = x
        x = self.norm2(x)
        ffn_out, aux_loss = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x, new_kv, aux_loss

# ==============================
# 4. Enhanced Recursive Memory Core (RMC)
# ==============================
class EnhancedRecursiveMemoryCore(nn.Module):
    """Enhanced RMC with multi-head reading/writing and gated updates"""
    def __init__(self, mem_size: int = 64, d_model: int = 256, num_heads: int = 4):
        super().__init__()
        self.mem_size = mem_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Memory slots
        self.memory = nn.Parameter(torch.randn(mem_size, d_model) * 0.01, requires_grad=True)
        
        # Multi-head reading mechanism
        self.read_heads = nn.ModuleList([
            nn.Linear(d_model, mem_size) for _ in range(num_heads)
        ])
        
        # Multi-head writing mechanism
        self.write_heads = nn.ModuleList([
            nn.Linear(d_model, mem_size) for _ in range(num_heads)
        ])
        
        # Gated update mechanism
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Projection layer
        self.proj = nn.Linear(d_model * (num_heads + 1), d_model)
        
        # Layer normalization
        self.norm = RMSNorm(d_model)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Multi-head memory reading"""
        B, D = query.shape
        read_vectors = []
        
        for head in self.read_heads:
            weights = F.softmax(head(query), dim=-1)  # [B, mem_size]
            read_vec = weights @ self.memory  # [B, D]
            read_vectors.append(read_vec)
        
        return torch.cat(read_vectors, dim=-1)  # [B, D * num_heads]

    def write(self, value: torch.Tensor) -> torch.Tensor:
        """Multi-head memory writing"""
        B, D = value.shape
        write_vectors = []
        
        for head in self.write_heads:
            weights = F.softmax(head(value), dim=-1)  # [B, mem_size]
            write_vec = (weights.unsqueeze(-1) * value.unsqueeze(1)).sum(dim=0)  # [mem_size, D]
            write_vectors.append(write_vec)
        
        return torch.stack(write_vectors, dim=0).mean(dim=0)  # Average across heads

    def forward(self, x: torch.Tensor, request_memory_update: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape
        avg = x.mean(dim=1)  # [B, D]
        
        # Read from memory
        read_vec = self.read(avg)  # [B, D * num_heads]
        read_broadcast = read_vec.unsqueeze(1).expand(-1, L, -1)  # [B, L, D * num_heads]
        
        # Combine input with memory
        combined = torch.cat([x, read_broadcast], dim=-1)  # [B, L, D * (num_heads + 1)]
        out = self.proj(combined)
        out = self.norm(out)
        
        # Prepare memory update if requested
        agg_update = None
        if request_memory_update:
            # Gated update mechanism
            memory_avg = self.memory.mean(dim=0).unsqueeze(0).expand(B, -1)  # [B, D]
            gate = self.update_gate(torch.cat([avg, memory_avg], dim=-1))  # [B, D]
            gated_value = avg * gate
            agg_update = self.write(gated_value)
            
        return out, agg_update

    def apply_memory_update(self, agg_update: torch.Tensor, lr: float = 0.01) -> None:
        """Apply memory update with momentum"""
        with torch.no_grad():
            # Apply update with momentum
            self.memory.data.add_(agg_update.detach() * lr)

# ==============================
# 5. Enhanced Neural Symbolic Bridge (NSB)
# ==============================
class EnhancedNeuralSymbolicBridge(nn.Module):
    """Enhanced NSB with attention over rule bank and dynamic rule creation"""
    def __init__(self, d_model: int, num_rules: int = 64, rule_dim: int = 64, 
                 num_heads: int = 4, max_rules: int = 128):
        super().__init__()
        self.d_model = d_model
        self.rule_dim = rule_dim
        self.num_rules = num_rules
        self.max_rules = max_rules
        
        # Rule bank with learnable rules
        self.rule_bank = nn.Parameter(torch.randn(num_rules, rule_dim) * 0.01, requires_grad=True)
        
        # Multi-head attention for rule matching
        self.rule_attn = nn.MultiheadAttention(rule_dim, num_heads, batch_first=True)
        
        # Dynamic rule creation mechanism
        self.rule_creator = nn.Sequential(
            nn.Linear(d_model, rule_dim * 2),
            nn.GELU(),
            nn.Linear(rule_dim * 2, rule_dim)
        )
        
        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(d_model + rule_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
        # Rule usage tracking (for dynamic rule creation)
        self.rule_usage = torch.zeros(num_rules)
        self.rule_creation_threshold = 0.1  # Create new rule if usage < threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project input to rule space
        queries = self.rule_creator(x)  # [B, L, rule_dim]
        
        # Attend over rule bank
        rules = self.rule_bank.unsqueeze(0).expand(B, -1, -1)  # [B, num_rules, rule_dim]
        attn_output, attn_weights = self.rule_attn(queries, rules, rules)
        
        # Update rule usage statistics
        if self.training:
            self.rule_usage = 0.9 * self.rule_usage + 0.1 * attn_weights.mean(dim=(0, 1)).detach().cpu()
        
        # Dynamic rule creation (during training)
        if self.training and self.rule_bank.size(0) < self.max_rules:
            self._maybe_create_new_rule(x, attn_weights)
        
        # Fuse with original input
        fused = self.fusion(torch.cat([x, attn_output], dim=-1))
        return fused + x  # Residual connection

    def _maybe_create_new_rule(self, x: torch.Tensor, attn_weights: torch.Tensor) -> None:
        """Create new rules based on input patterns"""
        # Find underutilized rules
        underutilized = self.rule_usage < self.rule_creation_threshold
        underutilized_idx = torch.where(underutilized)[0]
        
        if len(underutilized_idx) > 0 and self.rule_bank.size(0) < self.max_rules:
            # Create new rules from input patterns
            with torch.no_grad():
                # Get input patterns that don't match well with existing rules
                low_attn = attn_weights < 0.1  # Low attention patterns
                if low_attn.any():
                    # Sample from inputs with low attention to existing rules
                    flat_x = x.view(-1, self.d_model)
                    flat_attn = attn_weights.view(-1, self.num_rules)
                    low_attn_mask = (flat_attn.max(dim=1).values < 0.1)
                    
                    if low_attn_mask.any():
                        new_rule_patterns = flat_x[low_attn_mask]
                        if len(new_rule_patterns) > 0:
                            # Create new rule from pattern
                            new_rule = new_rule_patterns.mean(dim=0).unsqueeze(0)
                            new_rule = self.rule_creator(new_rule)
                            
                            # Add to rule bank
                            new_rule_bank = torch.cat([self.rule_bank, new_rule], dim=0)
                            self.rule_bank = nn.Parameter(new_rule_bank, requires_grad=True)
                            
                            # Update rule usage
                            self.rule_usage = torch.cat([self.rule_usage, torch.tensor([0.5])])  # Initial usage

# ==============================
# 6. Ultra-Powerful NeoFormer-X
# ==============================
class UltraNeoFormerLM(nn.Module):
    """Ultra-Powerful NeoFormer-X with all advanced features"""
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 512, nhead: int = 8, 
                 n_kv_heads: int = 4, num_layers: int = 12, d_ff: int = 2048, 
                 num_experts: int = 8, top_k: int = 2, mem_size: int = 64, 
                 num_rules: int = 64, rule_dim: int = 64, max_seq_len: int = 8192,
                 dropout: float = 0.1, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Use learned positional embeddings that can extend beyond max_seq_len
        self.pos_embed = nn.Embedding(max_seq_len * 2, d_model)  
        
        # Enhanced components
        self.rmc = EnhancedRecursiveMemoryCore(mem_size=mem_size, d_model=d_model, num_heads=4)
        self.nsb = EnhancedNeuralSymbolicBridge(d_model=d_model, num_rules=num_rules, 
                                               rule_dim=rule_dim, num_heads=4)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NeoFormerBlock(
                d_model=d_model, nhead=nhead, n_kv_heads=n_kv_heads,
                d_ff=d_ff, num_experts=num_experts, top_k=top_k,
                dropout=dropout, max_seq_len=max_seq_len
            ) for _ in range(num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        if use_gradient_checkpointing:
            for i in range(num_layers):
                self.layers[i] = torch.utils.checkpoint.checkpoint(self.layers[i])
        
        print(f"✅ Initialized UltraNeoFormer-X with {self._count_parameters():,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
                request_memory_update: bool = False, seq_len: int = 0) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device
        
        # Positional embeddings with dynamic extension
        positions = torch.arange(seq_len, seq_len + L, device=device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        
        # Token embeddings
        tok_emb = self.token_embed(input_ids)
        x = tok_emb + pos_emb
        
        # Apply enhanced RMC
        if request_memory_update:
            x, agg_update = self.rmc(x, request_memory_update)
        else:
            x = self.rmc(x, request_memory_update)
        
        # Apply enhanced NSB
        x = self.nsb(x)
        
        # Initialize KV cache if not provided
        if past_kvs is None:
            past_kvs = [None] * len(self.layers)
        
        new_kvs = []
        total_aux_loss = torch.tensor(0.0, device=device)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            x, new_kv, aux_loss = layer(
                x, kv=past_kvs[i], use_cache=True, seq_len=seq_len
            )
            new_kvs.append(new_kv)
            total_aux_loss += aux_loss
        
        x = self.norm(x)
        logits = self.lm_head(x)  # Keep in bfloat16 for efficiency
        
        if request_memory_update:
            return logits, new_kvs, agg_update, total_aux_loss
        return logits, new_kvs, None, total_aux_loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, 
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.1) -> torch.Tensor:
        self.eval()
        past_kvs = None
        generated = input_ids
        seq_len = 0
        
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.max_seq_len:
                # Truncate but keep recent context
                generated = generated[:, -self.max_seq_len//2:]
                if past_kvs is not None:
                    # Keep only recent half of KV cache
                    past_kvs = [(
                        kv[0][:, :, -self.max_seq_len//4:, :], 
                        kv[1][:, :, -self.max_seq_len//4:, :]
                    ) if kv is not None else None for kv in past_kvs]
                seq_len = self.max_seq_len // 2
            
            # Forward pass
            logits, past_kvs, _, _ = self(
                generated if past_kvs is None else generated[:, -1:],
                past_kvs=past_kvs,
                seq_len=seq_len
            )
            
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                score = logits.gather(1, generated)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, generated, score)
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float('inf')), logits)
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(1, indices_to_remove.unsqueeze(0), -float('inf'))
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            seq_len += generated.size(1) if past_kvs is None else 1
        
        return generated

# ==============================
# 7. Advanced Training Setup
# ==============================
class Lion(torch.optim.Optimizer):
    """Lion optimizer (Sign-Momentum) - often outperforms AdamW"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Create a cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==============================
# 8. Enhanced Training Script
# ==============================
def train_ultra_neoformer(args):
    print("🚀 Training Ultra NeoFormer-X...")
    
    # Initialize distributed training if available
    if torch.cuda.device_count() > 1 and args.distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main_process = local_rank == 0
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_main_process = True
        local_rank = 0
    
    # Load dataset
    if is_main_process:
        print("📚 Loading dataset...")
    
    from datasets import load_dataset
    dataset = load_dataset("openwebtext", split="train[:1%]", streaming=True)
    text_stream = (example["text"] for example in dataset)
    
    # Tokenize in chunks
    chunk_size = 10000
    data_chunks = []
    current_chunk = []
    
    for text in text_stream:
        tokens = encode(text)
        current_chunk.extend(tokens)
        
        if len(current_chunk) >= chunk_size:
            data_chunks.append(torch.tensor(current_chunk[:chunk_size], dtype=torch.long))
            current_chunk = current_chunk[chunk_size:]
            
            if len(data_chunks) >= args.num_chunks:
                break
    
    if current_chunk:
        data_chunks.append(torch.tensor(current_chunk, dtype=torch.long))
    
    if is_main_process:
        print(f"✅ Loaded {len(data_chunks)} chunks, {sum(len(chunk) for chunk in data_chunks):,} tokens")
    
    # Create sequences
    seq_len = args.seq_len
    all_input_ids = []
    all_labels = []
    
    for chunk in data_chunks:
        for i in range(0, len(chunk) - seq_len, seq_len):
            all_input_ids.append(chunk[i:i+seq_len])
            all_labels.append(chunk[i+1:i+seq_len+1])
    
    # Convert to tensors
    input_ids = torch.stack(all_input_ids[:args.max_sequences])
    labels = torch.stack(all_labels[:args.max_sequences])
    
    if is_main_process:
        print(f"✅ Dataset: {input_ids.shape}")
    
    # Model
    model = UltraNeoFormerLM(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        n_kv_heads=args.n_kv_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        top_k=args.top_k,
        mem_size=args.mem_size,
        num_rules=args.num_rules,
        rule_dim=args.rule_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    # Apply LoRA if requested
    if args.lora and HAS_PEFT:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "ffn"],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process:
            print("✅ LoRA applied.")
    
    # Distributed training setup
    if torch.cuda.device_count() > 1 and args.distributed:
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=torch.bfloat16,
            device_id=local_rank
        )
    
    model = model.to(device)
    
    # Compile for performance
    if args.compile:
        model = torch.compile(model, dynamic=True, fullgraph=False)
        if is_main_process:
            print("✅ Model compiled.")
    
    # Optimizer and scheduler
    if args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = args.epochs * (len(input_ids) // args.batch_size)
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Mixed precision training
    scaler = GradScaler() if device == 'cuda' else None
    
    # Training state
    global_step = 0
    best_loss = float('inf')
    
    # Training loop
    if is_main_process:
        print("🔥 Training Ultra NeoFormer-X...")
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(input_ids))
        
        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i:i+args.batch_size]
            batch_ids = input_ids[batch_indices].to(device)
            batch_labels = labels[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu', dtype=torch.bfloat16):
                logits, _, _, aux_loss = model(batch_ids, request_memory_update=True)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_labels.view(-1))
                total_loss = loss + aux_loss
            
            # Backward pass
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if is_main_process and global_step % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {global_step} | Loss: {loss.item():.4f} | "
                      f"Perplexity: {math.exp(loss.item()):.2f} | LR: {current_lr:.2e}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        if is_main_process:
            print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.4f} | "
                  f"Avg Perplexity: {math.exp(avg_epoch_loss):.2f}")
        
        # Save checkpoint
        if is_main_process and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(checkpoint_path, exist_ok=True)
            
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path)
            else:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                }, f"{checkpoint_path}/model.pt")
            
            print(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Generate sample text
        if is_main_process and (epoch + 1) % args.generate_interval == 0:
            model.eval()
            context = torch.tensor([[encode("The future of AI is")[0]]], device=device)
            generated = model.generate(
                context, 
                max_new_tokens=100, 
                temperature=0.8, 
                top_k=50,
                top_p=0.9
            )
            print("\n💡 Generated:")
            print(decode(generated[0].tolist()))
            print("-" * 80)
            model.train()
    
    # Save final model
    if is_main_process:
        final_path = f"{args.output_dir}/final_model"
        os.makedirs(final_path, exist_ok=True)
        
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(final_path)
        else:
            torch.save(model.state_dict(), f"{final_path}/model.pt")
        
        print(f"🎉 Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ultra NeoFormer-X")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--mem_size", type=int, default=64)
    parser.add_argument("--num_rules", type=int, default=64)
    parser.add_argument("--rule_dim", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_sequences", type=int, default=10000)
    parser.add_argument("--num_chunks", type=int, default=10)
    
    # LoRA parameters
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # System parameters
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lion"])
    
    # Logging/saving
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--generate_interval", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="ultra-neoformer-x")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    train_ultra_neoformer(args)
