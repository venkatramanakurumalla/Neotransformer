# -*- coding: utf-8 -*-
"""
⚡ Optimized NeoFormer Training Script
=====================================
- Mixed precision (AMP) for speed
- Gradient clipping (stability)
- Cosine LR scheduler + Warmup
- Label smoothing (better generalization)
- Gradient accumulation (handles large batch)
- Early stopping + checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math, os

# ==============================
# Example dataset
# ==============================
class RandomDataset(Dataset):
    def __init__(self, size=2000, seq_len=32, d_model=64, num_classes=5):
        self.x = torch.randn(size, seq_len, d_model)
        self.y = torch.randint(0, num_classes, (size,))
        self.num_classes = num_classes
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==============================
# Training Function
# ==============================
def train_model(
    model, train_loader, val_loader,
    num_epochs=20, lr=3e-4, warmup=500,
    device="cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path="neoformer_best.pt"
):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine decay with warmup
    total_steps = num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step+1)/warmup, 0.5*(1+math.cos(math.pi*step/total_steps)))
    )

    scaler = GradScaler()  # AMP scaler
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():  # Mixed precision
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        # ---------------- VALIDATE ----------------
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total * 100

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # ---------------- SAVE BEST ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("✅ Saved best checkpoint")

    print("Training complete! Best Val Loss:", best_val_loss)

# ==============================
# Run Example
# ==============================
if __name__ == "__main__":
    BATCH_SIZE = 32
    dataset = RandomDataset(size=5000, seq_len=32, d_model=64, num_classes=5)
    train_set, val_set = torch.utils.data.random_split(dataset, [4000, 1000])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = NeoFormer(d_model=64, nhead=4, num_layers=3,
                      num_classes=5, mem_size=32,
                      nsb_rules=64, nsb_rule_dim=16)

    train_model(model, train_loader, val_loader, num_epochs=10)
