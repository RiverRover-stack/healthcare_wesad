"""
Raspberry Pi Benchmark Script — Run this ON the Pi.

Measures:
  - Inference latency (ms)
  - Memory usage (MB)
  - CPU utilization (%)
  - Throughput (windows/sec)

Usage on RPi:
    pip install onnxruntime numpy psutil
    python benchmark_rpi.py --model micro_cnn.onnx --runs 100

Paper-ready output: prints a table you can paste directly into your paper.
"""

import time
import argparse
import numpy as np
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(model_path: str, n_runs: int = 100):
    import onnxruntime as ort

    # ── Session setup ──────────────────────────────────────────────────────────
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1   # single-thread = real-world wearable sim
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options)
    input_name = session.get_inputs()[0].name

    # ── Dummy input: 60s window @ 64Hz, 6 signals ─────────────────────────────
    dummy = np.random.randn(1, 6, 3840).astype(np.float32)

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # ── Memory before benchmark ────────────────────────────────────────────────
    mem_before = get_memory_mb()

    # ── Latency benchmark ─────────────────────────────────────────────────────
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    mem_after = get_memory_mb()

    # ── CPU utilization (over 5s continuous inference) ─────────────────────────
    cpu_readings = []
    t_end = time.time() + 5.0
    while time.time() < t_end:
        session.run(None, {input_name: dummy})
        cpu_readings.append(psutil.cpu_percent(interval=None))
    avg_cpu = np.mean(cpu_readings)

    # ── Model file size ────────────────────────────────────────────────────────
    model_size_kb = os.path.getsize(model_path) / 1024

    # ── Results ───────────────────────────────────────────────────────────────
    lat = np.array(latencies)
    throughput = 1000.0 / np.mean(lat)   # windows per second

    print("\n" + "="*55)
    print("  RASPBERRY PI INFERENCE BENCHMARK")
    print("="*55)
    print(f"  Model:            {os.path.basename(model_path)}")
    print(f"  Model size:       {model_size_kb:.1f} KB")
    print(f"  Runs:             {n_runs}")
    print("-"*55)
    print(f"  Latency mean:     {np.mean(lat):.2f} ms")
    print(f"  Latency std:      {np.std(lat):.2f} ms")
    print(f"  Latency p95:      {np.percentile(lat, 95):.2f} ms")
    print(f"  Latency min/max:  {np.min(lat):.2f} / {np.max(lat):.2f} ms")
    print(f"  Throughput:       {throughput:.1f} windows/sec")
    print("-"*55)
    print(f"  Memory (RSS):     {mem_after:.1f} MB  (+{mem_after - mem_before:.1f} MB during inference)")
    print(f"  CPU utilization:  {avg_cpu:.1f}%  (single thread)")
    print("="*55)
    print("\n  Window context: 60s window, 64Hz, 6 signals (3840 samples)")
    print(f"  Real-time factor: {np.mean(lat)/60000:.5f}x  (inference/window_duration)")
    print("  → Model is {:.0f}x faster than real-time\n".format(60000 / np.mean(lat)))

    # ── LaTeX-ready table row (copy into your paper) ───────────────────────────
    print("  LaTeX table row (paste into paper):")
    print(f"  MicroCNN-KD & {model_size_kb:.0f}KB & {np.mean(lat):.1f}ms & {mem_after:.0f}MB & {avg_cpu:.0f}\\% \\\\")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="micro_cnn.onnx", help="Path to .onnx file")
    parser.add_argument("--runs",  default=100, type=int,    help="Number of inference runs")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file '{args.model}' not found.")
        print("Copy micro_cnn.onnx from your main machine to this directory.")
        exit(1)

    run_benchmark(args.model, args.runs)
