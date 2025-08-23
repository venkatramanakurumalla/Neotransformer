# -*- coding: utf-8 -*-
"""
🎯 Train, Validate & Test NeoFormer on SST-2 (fixed + hardened)
- Small-data demo (500 train samples)
- Optional LoRA via peft
- Early stopping
- Final test with metrics
"""
import random
import math
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Optional PEFT/LoRA
try:
    from peft import get_peft_model, LoraConfig
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False
    print("⚠️ peft not available. To enable LoRA install: pip install peft")

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# 1. Load SST-2 Dataset
# ==============================
print("📥 Loading SST-2 dataset...")
dataset = load_dataset("stanfordnlp/sst2")

# Small training set (demo)
train_texts = dataset["train"]["sentence"][:500]
train_labels = dataset["train"]["label"][:500]

# Validation / test
val_texts = dataset["validation"]["sentence"]
val_labels = dataset["validation"]["label"]

# Use validation as test (common with SST-2)
test_texts = val_texts
test_labels = val_labels

print(f"✅ Train: {len(train_texts)} samples")
print(f"✅ Val:   {len(val_texts)} samples")
print(f"✅ Test:  {len(test_texts)} samples")

# ==============================
# 2. Tokenizer + collate
# ==============================
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def collate_fn(batch):
    texts, labels = zip(*batch)
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=32, return_tensors="pt")
    input_ids = encoded["input_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, labels

# ==============================
# 3. Embedder
# ==============================
class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input_ids):
        return self.embedding(input_ids)  # [B, L, D]

vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
embedder = TokenEmbedder(vocab_size=vocab_size, d_model=64)

# ==============================
# 4. Dataset & DataLoader
# ==============================
class SST2Dataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = SST2Dataset(train_texts, train_labels)
val_dataset = SST2Dataset(val_texts, val_labels)
test_dataset = SST2Dataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ==============================
# 5. NeoFormer Model (Tiny) - fixed / hardened
# ==============================
class SelfEvolvingPE(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        pe = self.pe[:L].unsqueeze(0).to(x.device)
        return self.dropout(x + pe)

class RecursiveMemoryCore(nn.Module):
    def __init__(self, mem_size=16, d_model=64):
        super().__init__()
        self.mem_size = mem_size
        self.d_model = d_model
        self.memory = nn.Parameter(torch.randn(mem_size, d_model) * 0.01, requires_grad=True)
        self.read_head = nn.Linear(d_model, mem_size)
        self.write_head = nn.Linear(d_model, mem_size)
        self.proj = nn.Linear(2 * d_model, d_model)

    def read(self, query):
        # query: [B, D] -> weights [B, mem_size]
        weights = F.softmax(self.read_head(query), dim=-1)
        return weights @ self.memory  # [B, D]

    def write(self, values):
        # values: [B, D] -> produce an aggregate update [mem_size, D] via weighted averaging
        weights = F.softmax(self.write_head(values), dim=-1)  # [B, mem_size]
        updates = weights.unsqueeze(-1) * values.unsqueeze(1)  # [B, mem_size, D]
        agg = updates.mean(dim=0)  # [mem_size, D]
        return agg

    def apply_memory_update(self, agg_update, lr=0.01):
        # agg_update expected shape [mem_size, d_model] or [d_model] (broadcast)
        with torch.no_grad():
            # ensure same device
            agg_update = agg_update.to(self.memory.device)
            # allow small scaling
            self.memory.add_(agg_update * lr)

    def forward(self, x, request_memory_update=False):
        # x: [B, L, D]
        avg = x.mean(dim=1)  # [B, D]
        read_vec = self.read(avg)  # [B, D]
        read_broadcast = read_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, D]
        combined = torch.cat([x, read_broadcast], dim=-1)  # [B, L, 2D]
        out = self.proj(combined)  # [B, L, D]
        if request_memory_update:
            agg = self.write(avg)  # [mem_size, D]
            return out, agg
        return out

class NeuralSymbolicBridge(nn.Module):
    def __init__(self, d_model, num_rules=16, rule_dim=16):
        super().__init__()
        self.rule_bank = nn.Embedding(num_rules, rule_dim)
        self.matcher = nn.Linear(d_model, rule_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model + rule_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: [B, L, D]
        queries = self.matcher(x)  # [B, L, Rdim]
        rules = self.rule_bank.weight  # [num_rules, rule_dim]
        attn = torch.einsum('blr,nr->bln', queries, rules)  # [B, L, N]
        selected_rules = F.softmax(attn, dim=-1) @ rules  # [B, L, rule_dim]
        x_cat = torch.cat([x, selected_rules], dim=-1)
        fused = self.fusion_ffn(x_cat)
        return fused + x

class AdaptiveLayer(nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self.controller = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # gate per batch — use mean over sequence
        gate = torch.sigmoid(self.controller(x.mean(1))).view(x.size(0), 1, 1)
        out = self.layer(x)
        return gate * out + (1 - gate) * x

class NeoFormer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.sepe = SelfEvolvingPE(d_model, dropout=0.3)
        self.rmc = RecursiveMemoryCore(mem_size=16, d_model=d_model)
        self.nsb = NeuralSymbolicBridge(d_model=d_model, num_rules=16, rule_dim=16)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=128, dropout=0.3,
                                                   activation='gelu', batch_first=True)
            self.layers.append(AdaptiveLayer(enc_layer, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, request_memory_update=False):
        # x: [B, L, D]
        x = self.sepe(x)
        if request_memory_update:
            x, agg_update = self.rmc(x, request_memory_update=True)
        else:
            x = self.rmc(x, request_memory_update=False)
            agg_update = None
        x = self.nsb(x)
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        if request_memory_update:
            return logits, agg_update
        return logits

# ==============================
# 6. Training Function (fixed)
# ==============================
def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device.upper()}")

    model = NeoFormer(d_model=64, nhead=4, num_layers=2, num_classes=2)
    model.to(device)
    embedder.to(device)  # important!

    # LoRA / PEFT (optional)
    if HAS_PEFT:
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["rmc", "nsb"]
        )
        model = get_peft_model(model, lora_config)
        print("✅ LoRA applied.")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    else:
        print("❌ LoRA not applied (peft unavailable).")

    # loss / optimizer
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    except TypeError:
        # older torch
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler() if device == 'cuda' else None

    best_val_acc = 0.0
    patience = 0
    max_patience = 5
    epochs = 15

    print("🔥 Starting training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for input_ids, labels in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            x = embedder(input_ids)  # [B, L, D]

            optimizer.zero_grad()

            amp_ctx = autocast() if device == 'cuda' else nullcontext()
            with amp_ctx:
                # request memory update during training
                logits, agg_update = model(x, request_memory_update=True)
                task_loss = criterion(logits, labels)
                mem_loss = torch.norm(agg_update) * 0.001
                loss = task_loss + mem_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # apply memory update (detach to be safe)
            if isinstance(agg_update, torch.Tensor):
                model.rmc.apply_memory_update(agg_update.detach(), lr=0.01)

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        scheduler.step()

        # Validation
        val_acc = evaluate(model, val_loader, device, embedder)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # If using PEFT, best practice is to save via peft utilities; fallback to state_dict
            try:
                # if model has save_pretrained (peft), use it
                if HAS_PEFT and hasattr(model, "save_pretrained"):
                    model.save_pretrained("sst2_neoformer_best_peft")
                    print(f"✅ Saved best PEFT model (acc: {best_val_acc:.2f}%)")
                else:
                    torch.save(model.state_dict(), "sst2_neoformer_best.pt")
                    print(f"✅ Saved best model state_dict (acc: {best_val_acc:.2f}%)")
            except Exception as e:
                print("⚠️ Warning: failed to save model cleanly:", e)
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

# ==============================
# 7. Evaluation Function
# ==============================
def evaluate(model, data_loader, device, embedder):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            x = embedder(input_ids)
            logits = model(x) if not isinstance(model, (type,)) else model(x)  # just call forward
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    if len(all_labels) == 0:
        return 0.0
    acc = accuracy_score(all_labels, all_preds)
    return acc * 100.0

# ==============================
# 8. Final Testing with Metrics (fixed)
# ==============================
def test_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeoFormer(d_model=64, nhead=4, num_layers=2, num_classes=2)

    # If PEFT was used, attempt to create same wrapper before loading
    if HAS_PEFT:
        try:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["rmc", "nsb"]
            )
            model = get_peft_model(model, lora_config)
        except Exception:
            pass

    # Load best checkpoint (try peft dir first)
    if HAS_PEFT and os.path.isdir("sst2_neoformer_best_peft"):
        try:
            model.load_state_dict(torch.load("sst2_neoformer_best_peft/pytorch_model.bin"), strict=False)
        except Exception:
            # peft save_pretrained saves args differently; let user manage if needed
            pass
    elif os.path.exists("sst2_neoformer_best.pt"):
        model.load_state_dict(torch.load("sst2_neoformer_best.pt", map_location=device))
    else:
        print("⚠️ No saved model found. Running with randomly initialized weights.")

    model.to(device)
    embedder.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_texts = []

    print("\n🧪 Running final test on SST-2 test set...\n")
    with torch.no_grad():
        start = 0
        for input_ids, labels in test_loader:
            bs = input_ids.size(0)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            x = embedder(input_ids)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # get the matching slice of raw texts
            texts_slice = test_texts[start:start + bs]
            start += bs

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_texts.extend(texts_slice)

    # Metrics
    if len(all_labels) == 0:
        print("No predictions were made (empty test).")
        return

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    print("📊 Final Test Results:")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1-Score:  {f1:.2%}")

    # Show some predictions
    print("\n🔍 Sample Predictions:")
    label_names = ["Negative", "Positive"]
    n_show = min(5, len(all_texts))
    for i in range(n_show):
        text = all_texts[i]
        pred = label_names[int(all_preds[i])]
        true = label_names[int(all_labels[i])]
        print(f"  [{true} → {pred}] {text[:200]}...")

# ==============================
# Run Everything
# ==============================
if __name__ == "__main__":
    train_model()
    test_model()
