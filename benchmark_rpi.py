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

Paper-ready output: prints a table you can paste directly into your paper, and
writes rpi_benchmark.json + rpi_benchmark.txt next to the script so the run has
a saved record (previously every Pi number in the paper was hand-transcribed
from a console with no log at all).
"""

import time
import argparse
import json
import threading
import numpy as np
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(model_path: str, n_runs: int = 100, cpu_window_sec: float = 1.0):
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

    # ── CPU utilization: THIS process only, over one real fixed window ─────────
    # The old version called psutil.cpu_percent(interval=None) inside the
    # inference loop: that's system-wide (every process, all cores), unprimed
    # (a process's/system's first cpu_percent() call always returns 0.0 -- there
    # is no prior sample to diff against), and sampled at ~ms intervals where
    # the value is quantised OS-tick noise. Fixed by measuring this process
    # specifically (psutil.Process().cpu_percent), primed with a throwaway call,
    # over one genuine cpu_window_sec-second window. That call blocks the
    # calling thread, so a background thread keeps running inference during the
    # window -- otherwise there is no work for it to measure.
    # NOTE: per-process, NOT divided by core count -- 100% means one full core
    # saturated (matches intra_op_num_threads=1 above); it is not a system-wide
    # or per-core-normalised figure.
    stop_event = threading.Event()
    cpu_window_runs = [0]

    def _inference_worker():
        while not stop_event.is_set():
            session.run(None, {input_name: dummy})
            cpu_window_runs[0] += 1

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # prime -- discard the meaningless first reading
    worker = threading.Thread(target=_inference_worker, daemon=True)
    worker.start()
    cpu_percent = process.cpu_percent(interval=cpu_window_sec)  # blocks cpu_window_sec seconds
    stop_event.set()
    worker.join(timeout=5.0)

    # ── Model file size ────────────────────────────────────────────────────────
    model_size_kb = os.path.getsize(model_path) / 1024

    # ── Results ───────────────────────────────────────────────────────────────
    lat = np.array(latencies)
    throughput = 1000.0 / np.mean(lat)   # windows per second

    results = {
        'model_path': os.path.basename(model_path),
        'model_size_kb': float(model_size_kb),
        'n_runs': n_runs,
        'latency_mean_ms': float(np.mean(lat)),
        'latency_std_ms': float(np.std(lat)),
        'latency_p95_ms': float(np.percentile(lat, 95)),
        'latency_min_ms': float(np.min(lat)),
        'latency_max_ms': float(np.max(lat)),
        'throughput_windows_per_sec': float(throughput),
        'memory_rss_mb': float(mem_after),
        'memory_delta_mb': float(mem_after - mem_before),
        'cpu_percent': float(cpu_percent),
        'cpu_percent_method': (
            'psutil.Process(pid).cpu_percent(interval=%.1f) -- per-process, '
            'NOT divided by core count (100%% = one full core saturated); '
            'not system-wide.' % cpu_window_sec
        ),
        'cpu_window_sec': cpu_window_sec,
        'cpu_window_inference_runs': cpu_window_runs[0],
        'window_sec': 60,
        'sample_rate_hz': 64,
        'n_channels': 6,
        'onnx_intra_op_num_threads': 1,
    }

    print("\n" + "="*55)
    print("  RASPBERRY PI INFERENCE BENCHMARK")
    print("="*55)
    print(f"  Model:            {results['model_path']}")
    print(f"  Model size:       {results['model_size_kb']:.1f} KB")
    print(f"  Runs:             {results['n_runs']}")
    print("-"*55)
    print(f"  Latency mean:     {results['latency_mean_ms']:.2f} ms")
    print(f"  Latency std:      {results['latency_std_ms']:.2f} ms")
    print(f"  Latency p95:      {results['latency_p95_ms']:.2f} ms")
    print(f"  Latency min/max:  {results['latency_min_ms']:.2f} / {results['latency_max_ms']:.2f} ms")
    print(f"  Throughput:       {results['throughput_windows_per_sec']:.1f} windows/sec")
    print("-"*55)
    print(f"  Memory (RSS):     {results['memory_rss_mb']:.1f} MB  (+{results['memory_delta_mb']:.1f} MB during inference)")
    print(f"  CPU utilization:  {results['cpu_percent']:.1f}%  (this process, single thread, "
          f"NOT core-normalised, {cpu_window_sec:.1f}s window, "
          f"{results['cpu_window_inference_runs']} inferences during window)")
    print("="*55)
    print("\n  Window context: 60s window, 64Hz, 6 signals (3840 samples)")
    print(f"  Real-time factor: {results['latency_mean_ms']/60000:.5f}x  (inference/window_duration)")
    print("  -> Model is {:.0f}x faster than real-time\n".format(60000 / results['latency_mean_ms']))

    # ── LaTeX-ready table row (copy into your paper) ───────────────────────────
    print("  LaTeX table row (paste into paper):")
    print(f"  MicroCNN-KD & {results['model_size_kb']:.0f}KB & {results['latency_mean_ms']:.1f}ms & "
          f"{results['memory_rss_mb']:.0f}MB & {results['cpu_percent']:.0f}\\% \\\\")
    print()

    # ── Persist ──────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(model_path)) or '.'
    json_path = os.path.join(out_dir, 'rpi_benchmark.json')
    txt_path = os.path.join(out_dir, 'rpi_benchmark.txt')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {results['model_path']} ({results['model_size_kb']:.1f} KB)\n")
        f.write(f"Runs: {results['n_runs']}\n")
        f.write(f"Latency: mean={results['latency_mean_ms']:.2f}ms std={results['latency_std_ms']:.2f}ms "
                f"p95={results['latency_p95_ms']:.2f}ms min={results['latency_min_ms']:.2f}ms "
                f"max={results['latency_max_ms']:.2f}ms\n")
        f.write(f"Throughput: {results['throughput_windows_per_sec']:.1f} windows/sec\n")
        f.write(f"Memory RSS: {results['memory_rss_mb']:.1f} MB (delta +{results['memory_delta_mb']:.1f} MB)\n")
        f.write(f"CPU: {results['cpu_percent']:.1f}% -- {results['cpu_percent_method']}\n")
        f.write(f"LaTeX row: MicroCNN-KD & {results['model_size_kb']:.0f}KB & "
                f"{results['latency_mean_ms']:.1f}ms & {results['memory_rss_mb']:.0f}MB & "
                f"{results['cpu_percent']:.0f}\\% \\\\\n")
    print(f"  Saved -> {json_path}")
    print(f"  Saved -> {txt_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="micro_cnn.onnx", help="Path to .onnx file")
    parser.add_argument("--runs",  default=100, type=int,    help="Number of inference runs")
    parser.add_argument("--cpu-window", default=1.0, type=float,
                        help="Seconds to measure process CPU% over (default 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file '{args.model}' not found.")
        print("Copy micro_cnn.onnx from your main machine to this directory.")
        exit(1)

    run_benchmark(args.model, args.runs, args.cpu_window)
