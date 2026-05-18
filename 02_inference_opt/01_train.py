"""
训练英→德翻译模型，保存最佳 checkpoint
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_data, PAD
from model import TranslationTransformer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("加载数据...", flush=True)
    train_loader, val_loader, src_vocab, tgt_vocab = load_data(batch_size=64, max_len=64)
    print(f"词表: src={len(src_vocab)}, tgt={len(tgt_vocab)}", flush=True)
    print(f"样本: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}", flush=True)

    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=1024, dropout=0.1, pad_id=PAD,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val_loss = float("inf")
    num_epochs = 10

    print(f"\n开始训练 {num_epochs} epochs...", flush=True)
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "val_loss": val_loss,
            }, CHECKPOINT_PATH)
            improved = " *"

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.1e} | "
              f"Time: {elapsed:.1f}s{improved}", flush=True)

    print(f"\n训练完成! 最佳 val_loss: {best_val_loss:.4f}")
    print(f"模型已保存: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
