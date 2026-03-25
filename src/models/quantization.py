"""
Post-Training Quantization Module

Applies INT8 dynamic quantization to reduce model size for edge deployment.
No retraining or calibration data required for dynamic quantization.

Dynamic quantization:
    - Weights are quantized to INT8 at load time (stored as int8)
    - Activations are quantized on-the-fly at runtime (per-batch, per-tensor)
    - Supported layers: nn.Linear, nn.LSTM, nn.Conv1d (via backend)
    - Typical size reduction: 3-4x over FP32

Usage:
    from src.models.quantization import quantize_model, compare_fp32_vs_int8

    # Quantize a trained model
    q_model = quantize_model(trained_micro_cnn)

    # Measure size and accuracy changes
    report = compare_fp32_vs_int8(trained_micro_cnn, input_shape=(6, 3840))
    print(report)
"""

import io
import torch
import torch.nn as nn
from typing import Dict, Tuple


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply INT8 dynamic quantization and return the quantized model.

    The original model is NOT modified — a new quantized copy is returned.
    Dynamic quantization is the simplest approach: no calibration data needed,
    and it works well for inference-only deployment.

    Args:
        model:  A trained nn.Module (should be in eval mode).

    Returns:
        A new nn.Module with INT8 quantized Linear and LSTM layers,
        and INT8 quantized Conv1d layers where the backend supports it.
    """
    model.eval()
    model_cpu = model.cpu()

    # Layers to quantize — Conv1d is included but may fall back to float
    # on some platforms; Linear and LSTM always quantize successfully.
    layers_to_quantize = {nn.Linear, nn.LSTM, nn.Conv1d}

    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec=layers_to_quantize,
        dtype=torch.qint8,
    )
    return quantized


def model_size_kb(model: nn.Module) -> float:
    """Serialise model state to BytesIO and return size in KB."""
    buf = io.BytesIO()
    # quantized models may not support state_dict saving in all versions
    try:
        torch.save(model.state_dict(), buf)
    except Exception:
        torch.save(model, buf)
    return buf.tell() / 1024.0


def compare_fp32_vs_int8(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (6, 3840),
) -> Dict:
    """
    Quantize a model and report size and latency changes.

    Does NOT measure accuracy change — that requires running LOSO
    evaluation with the quantized model (see train_students.py).

    Args:
        model:        Trained FP32 model.
        input_shape:  Signal tensor shape WITHOUT batch dim.

    Returns:
        Dict with keys: fp32_kb, int8_kb, size_reduction_x,
                        fp32_latency_ms, int8_latency_ms, speedup_x
    """
    import time
    import numpy as np

    model.eval()

    # FP32 baseline
    fp32_kb = model_size_kb(model)
    dummy   = torch.zeros(1, *input_shape)
    with torch.no_grad():
        for _ in range(5):           # warmup
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(50):
            model(dummy)
        fp32_ms = (time.perf_counter() - t0) / 50 * 1000

    # INT8 quantized
    q_model = quantize_model(model)
    int8_kb = model_size_kb(q_model)
    with torch.no_grad():
        for _ in range(5):           # warmup
            q_model(dummy)
        t0 = time.perf_counter()
        for _ in range(50):
            q_model(dummy)
        int8_ms = (time.perf_counter() - t0) / 50 * 1000

    size_reduction = fp32_kb / max(int8_kb, 1e-6)
    speedup        = fp32_ms / max(int8_ms, 1e-6)

    return {
        'fp32_kb':          round(fp32_kb, 2),
        'int8_kb':          round(int8_kb, 2),
        'size_reduction_x': round(size_reduction, 2),
        'fp32_latency_ms':  round(fp32_ms, 3),
        'int8_latency_ms':  round(int8_ms, 3),
        'speedup_x':        round(speedup, 2),
    }


def print_quantization_report(model_name: str, report: Dict) -> None:
    """Pretty-print the output of compare_fp32_vs_int8."""
    print(f"\n  Quantization report: {model_name}")
    print(f"    Size:    FP32={report['fp32_kb']:.1f}KB"
          f"  INT8={report['int8_kb']:.1f}KB"
          f"  ({report['size_reduction_x']:.1f}x reduction)")
    print(f"    Latency: FP32={report['fp32_latency_ms']:.1f}ms"
          f"  INT8={report['int8_latency_ms']:.1f}ms"
          f"  ({report['speedup_x']:.1f}x speedup)")
