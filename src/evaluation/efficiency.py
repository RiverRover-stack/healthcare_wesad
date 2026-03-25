"""
Efficiency Metrics Module

Measures the resource cost of each model — the key differentiator of this
paper over prior WESAD work (which only reports accuracy metrics).

Metrics reported:
    params      Number of trainable parameters
    size_kb     Model file size in KB (FP32 float32 weights)
    latency_ms  Mean CPU inference time per single window (ms)
    flops       Multiply-accumulate operations per inference (if thop installed)

Usage:
    from src.evaluation.efficiency import get_efficiency_report, print_efficiency_table

    report = get_efficiency_report(model, input_shape=(6, 3840))
    print_efficiency_table({'Teacher': teacher_report, 'MicroCNN': micro_report})
"""

import io
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    """
    Measure FP32 model size by serialising state_dict to an in-memory buffer.
    More accurate than estimating from param count because it includes BN buffers.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def measure_latency_ms(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = 'cpu',
) -> float:
    """
    Measure mean single-sample inference latency on CPU.

    Args:
        model:        The model to benchmark.
        input_shape:  Shape WITHOUT the batch dimension, e.g. (6, 3840).
        n_warmup:     Warmup iterations (not included in timing).
        n_runs:       Timed iterations.
        device:       'cpu' (edge simulation) or 'cuda'.

    Returns:
        Mean latency per sample in milliseconds.
    """
    model = model.to(device)
    model.eval()
    # Single sample (batch=1) mimics edge device inference
    dummy = torch.zeros(1, *input_shape).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times))


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Optional[int]:
    """
    Count multiply-accumulate operations (MACs) for one forward pass.

    Uses the `thop` library if available (pip install thop).
    Returns None if thop is not installed rather than raising.

    Note: thop counts MACs; FLOPs ≈ 2 * MACs for dense layers.
    """
    try:
        from thop import profile
        dummy = torch.zeros(1, *input_shape)
        model_cpu = model.cpu()
        macs, _ = profile(model_cpu, inputs=(dummy,), verbose=False)
        return int(macs)
    except ImportError:
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Composite report
# ─────────────────────────────────────────────────────────────────────────────

def get_efficiency_report(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (6, 3840),
) -> Dict:
    """
    Run all efficiency measurements for one model.

    Returns a dict with keys:
        params      int    — trainable parameter count
        size_kb     float  — FP32 model size in KB
        latency_ms  float  — mean CPU latency per sample
        flops       int|None — MACs if thop available, else None
    """
    model.eval()
    return {
        'params':      count_parameters(model),
        'size_kb':     round(model_size_kb(model), 2),
        'latency_ms':  round(measure_latency_ms(model, input_shape), 3),
        'flops':       count_flops(model, input_shape),
    }


def run_all_efficiency_benchmarks(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...] = (6, 3840),
) -> Dict[str, Dict]:
    """
    Benchmark a dict of models and return a nested results dict.

    Args:
        models:  {'ModelName': model_instance, ...}
    Returns:
        {'ModelName': {'params': ..., 'size_kb': ..., 'latency_ms': ..., 'flops': ...}, ...}
    """
    results = {}
    for name, model in models.items():
        print(f"  Benchmarking {name} ...")
        results[name] = get_efficiency_report(model, input_shape)
        r = results[name]
        flops_str = f"{r['flops']:,}" if r['flops'] else "N/A (install thop)"
        print(f"    params={r['params']:,}  size={r['size_kb']:.1f}KB"
              f"  latency={r['latency_ms']:.1f}ms  flops={flops_str}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_efficiency_table(results: Dict[str, Dict]) -> None:
    """Print a formatted efficiency comparison table to stdout."""
    header = f"{'Model':<22} {'Params':>10} {'Size(KB)':>10} {'Latency(ms)':>13} {'FLOPs':>14}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name, r in results.items():
        flops = f"{r['flops']:,}" if r['flops'] else "N/A"
        print(f"{name:<22} {r['params']:>10,} {r['size_kb']:>10.1f}"
              f" {r['latency_ms']:>13.1f} {flops:>14}")
    print("─" * len(header))
