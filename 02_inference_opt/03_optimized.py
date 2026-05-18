"""
03_optimized.py
多级优化推理 + 与基线速度对比
Level 1: BF16 半精度
Level 2: torch.compile (图模式)
Level 3: 定制 CUDA 算子 + BF16
"""

import os
import sys
import torch
import torch.nn as nn
import time
import math

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_data, PAD
from model import TranslationTransformer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")


def load_model(device, dtype=torch.float32):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = TranslationTransformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=1024, dropout=0.1, pad_id=PAD,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def benchmark(model, src, tgt_in, warmup=50, iters=200):
    device = src.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(src, tgt_in)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(src, tgt_in)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        peak_mem = (torch.cuda.max_memory_allocated(device) - mem_before) / 1024 / 1024
    return avg_ms, peak_mem


def test_correctness(model, src, tgt_in, tgt_out, tgt_vocab, label, dtype=torch.float32):
    """验证优化后模型的数值精度"""
    with torch.no_grad():
        logits = model(src, tgt_in)
        pred = logits.argmax(-1)
        pred_str = tgt_vocab.decode(pred[0].tolist())
    ref_str = tgt_vocab.decode(tgt_out[0].tolist())
    match = "OK" if pred_str == ref_str else "DIFF"
    print(f"  [{label}] 预测: {pred_str[:60]}  ({match})")


# ======================================================================
# Level 1: BF16 半精度推理
# ======================================================================
def run_level1(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results):
    print("\n" + "=" * 60)
    print("Level 1: BF16 半精度推理")
    print("=" * 60)

    model = load_model(device, dtype=torch.bfloat16)
    model.pos_enc.pe = model.pos_enc.pe.to(torch.bfloat16)
    # src/tgt 是 token ids (long), 保持不变, 模型内部 embedding 后自动变为 bf16
    test_correctness(model, src, tgt_in, tgt_out, tgt_vocab, "BF16", torch.bfloat16)

    results = {}
    print(f"\n{'Batch':>6} | {'延迟(ms)':>10} | {'加速比':>8} | {'显存(MB)':>10} | {'显存节省':>8}")
    print("-" * 60)
    for bs in [1, 4, 8, 16, 32]:
        src_b = src.repeat(bs, 1)
        tgt_b = tgt_in.repeat(bs, 1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            avg_ms, peak_mem = benchmark(model, src_b, tgt_b)
        base_ms = baseline_results[bs]["latency"]
        speedup = base_ms / avg_ms
        mem_save = 1 - peak_mem / baseline_results[bs]["memory"] if baseline_results[bs]["memory"] > 0 else 0
        results[bs] = {"latency": avg_ms, "memory": peak_mem, "speedup": speedup}
        print(f"{bs:>6} | {avg_ms:>10.3f} | {speedup:>7.2f}x | {peak_mem:>10.1f} | {mem_save:>7.1%}")
    return results


# ======================================================================
# Level 2: torch.compile (图模式)
# ======================================================================
def run_level2(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results):
    print("\n" + "=" * 60)
    print("Level 2: torch.compile (图模式)")
    print("=" * 60)

    model = load_model(device)
    print("  编译模型中 (首次较慢)...")
    compiled_model = torch.compile(model, mode="reduce-overhead")

    test_correctness(compiled_model, src, tgt_in, tgt_out, tgt_vocab, "compile")

    results = {}
    print(f"\n{'Batch':>6} | {'延迟(ms)':>10} | {'加速比':>8} | {'显存(MB)':>10}")
    print("-" * 55)
    for bs in [1, 4, 8, 16, 32]:
        src_b = src.repeat(bs, 1)
        tgt_b = tgt_in.repeat(bs, 1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            avg_ms, peak_mem = benchmark(compiled_model, src_b, tgt_b, warmup=30, iters=200)
        base_ms = baseline_results[bs]["latency"]
        speedup = base_ms / avg_ms
        results[bs] = {"latency": avg_ms, "memory": peak_mem, "speedup": speedup}
        print(f"{bs:>6} | {avg_ms:>10.3f} | {speedup:>7.2f}x | {peak_mem:>10.1f}")
    return results


# ======================================================================
# Level 3: 定制 CUDA 算子 + BF16
# ======================================================================
def run_level3(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results):
    print("\n" + "=" * 60)
    print("Level 3: 定制 CUDA 算子 (FusedLayerNorm + FusedSoftmaxMask) + BF16")
    print("=" * 60)

    from custom_ops import fused_layernorm, fused_softmax_mask

    model = load_model(device, dtype=torch.bfloat16)
    model.pos_enc.pe = model.pos_enc.pe.to(torch.bfloat16)

    # 用定制算子替换模型的 LayerNorm
    # monkey-patch transformer 内部所有 LayerNorm 为 fused 版本
    for name, module in model.transformer.named_modules():
        if isinstance(module, nn.LayerNorm):
            weight = module.weight
            bias = module.bias
            eps = module.eps
            module.forward = lambda x, _w=weight, _b=bias, _e=eps: fused_layernorm(x, _w, _b, _e)

    # 替换原始 forward, 跳过 pos_enc 的 dropout (推理时无效)
    def _patched_forward(src, tgt):
        scale = math.sqrt(model.d_model)
        src_emb = model.src_emb(src) * scale + model.pos_enc.pe[:, :src.size(1)]
        tgt_emb = model.tgt_emb(tgt) * scale + model.pos_enc.pe[:, :tgt.size(1)]
        tgt_mask = model._make_tgt_mask(tgt)
        out = model.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return model.fc_out(out)

    model.forward = _patched_forward

    test_correctness(model, src, tgt_in, tgt_out, tgt_vocab, "FusedOps+BF16", torch.bfloat16)

    results = {}
    print(f"\n{'Batch':>6} | {'延迟(ms)':>10} | {'加速比':>8} | {'显存(MB)':>10}")
    print("-" * 55)
    for bs in [1, 4, 8, 16, 32]:
        src_b = src.repeat(bs, 1)
        tgt_b = tgt_in.repeat(bs, 1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            avg_ms, peak_mem = benchmark(model, src_b, tgt_b)
        base_ms = baseline_results[bs]["latency"]
        speedup = base_ms / avg_ms
        results[bs] = {"latency": avg_ms, "memory": peak_mem, "speedup": speedup}
        print(f"{bs:>6} | {avg_ms:>10.3f} | {speedup:>7.2f}x | {peak_mem:>10.1f}")
    return results


# ======================================================================
# 汇总对比
# ======================================================================
def print_summary(baseline, l1, l2, l3):
    print("\n" + "=" * 80)
    print("性能汇总对比 (延迟 ms, 加速比 vs FP32 Eager 基线)")
    print("=" * 80)
    header = f"{'Batch':>6}"
    header += f" | {'基线':>10}"
    header += f" | {'BF16':>10} {'加速':>6}"
    header += f" | {'compile':>10} {'加速':>6}"
    header += f" | {'Fused+BF16':>10} {'加速':>6}"
    print(header)
    print("-" * 80)

    for bs in [1, 4, 8, 16, 32]:
        line = f"{bs:>6}"
        line += f" | {baseline[bs]['latency']:>10.3f}"
        line += f" | {l1[bs]['latency']:>10.3f} {l1[bs]['speedup']:>5.2f}x"
        line += f" | {l2[bs]['latency']:>10.3f} {l2[bs]['speedup']:>5.2f}x"
        line += f" | {l3[bs]['latency']:>10.3f} {l3[bs]['speedup']:>5.2f}x"
        print(line)

    print("\n" + "=" * 80)
    print("显存对比 (MB)")
    print("=" * 80)
    header = f"{'Batch':>6}"
    header += f" | {'基线':>10}"
    header += f" | {'BF16':>10} {'节省':>6}"
    header += f" | {'compile':>10} {'节省':>6}"
    header += f" | {'Fused+BF16':>10} {'节省':>6}"
    print(header)
    print("-" * 80)

    for bs in [1, 4, 8, 16, 32]:
        base_mem = baseline[bs]["memory"]
        line = f"{bs:>6}"
        line += f" | {base_mem:>10.1f}"
        line += f" | {l1[bs]['memory']:>10.1f} {(1 - l1[bs]['memory']/base_mem):>5.0%}" if base_mem > 0 else f" | {'N/A':>10} {'N/A':>6}"
        line += f" | {l2[bs]['memory']:>10.1f} {(1 - l2[bs]['memory']/base_mem):>5.0%}" if base_mem > 0 else f" | {'N/A':>10} {'N/A':>6}"
        line += f" | {l3[bs]['memory']:>10.1f} {(1 - l3[bs]['memory']/base_mem):>5.0%}" if base_mem > 0 else f" | {'N/A':>10} {'N/A':>6}"
        print(line)


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    _, val_loader, src_vocab, tgt_vocab = load_data(batch_size=1, max_len=64)
    src, tgt_in, tgt_out = next(iter(val_loader))
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

    # 先运行基线获取基准数据
    print("加载基线模型...")
    baseline_model = load_model(device)
    baseline_results = {}
    print(f"\n{'Batch':>6} | {'延迟(ms)':>10} | {'显存(MB)':>10}")
    print("-" * 40)
    for bs in [1, 4, 8, 16, 32]:
        src_b = src.repeat(bs, 1)
        tgt_b = tgt_in.repeat(bs, 1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            avg_ms, peak_mem = benchmark(baseline_model, src_b, tgt_b)
        baseline_results[bs] = {"latency": avg_ms, "memory": peak_mem}
        print(f"{bs:>6} | {avg_ms:>10.3f} | {peak_mem:>10.1f}")
    del baseline_model
    torch.cuda.empty_cache()

    # 逐级优化
    l1 = run_level1(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results)
    torch.cuda.empty_cache()

    l2 = run_level2(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results)
    torch.cuda.empty_cache()

    l3 = run_level3(device, src, tgt_in, tgt_out, tgt_vocab, baseline_results)

    # 汇总
    print_summary(baseline_results, l1, l2, l3)


if __name__ == "__main__":
    main()
