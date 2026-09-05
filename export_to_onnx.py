"""
Export MicroCNN to ONNX for Raspberry Pi deployment.

Run this on your main machine AFTER training students:
    python export_to_onnx.py

Exports from MODELS_DIR/MicroCNN_distilled_loso_S2.pt -- i.e. whatever
WESAD_OUTPUT_DIR points at for this run -- so it always uses the checkpoint
from the run that produced the paper's numbers, not a stale one from a
different run.

Output: <MODELS_DIR>/micro_cnn.onnx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import MODELS_DIR
from models.student import MicroCNN

CHECKPOINT = MODELS_DIR / "MicroCNN_distilled_loso_S2.pt"
OUTPUT_PATH = MODELS_DIR / "micro_cnn.onnx"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

model = MicroCNN(in_channels=6, num_classes=2)
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

# Handle checkpoint format: dict with nested state_dict vs bare state_dict
if isinstance(state, dict):
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in state:
            model.load_state_dict(state[key])
            break
    else:
        model.load_state_dict(state)
else:
    model = state

model.eval()

# ── Export to ONNX ─────────────────────────────────────────────────────────────
# Input shape: (batch=1, channels=6, samples=3840) — 60s @ 64Hz
dummy_input = torch.zeros(1, 6, 3840)

torch.onnx.export(
    model,
    dummy_input,
    str(OUTPUT_PATH),
    export_params=True,
    opset_version=11,           # RPi onnxruntime supports opset 11
    input_names=["signals"],
    output_names=["logits"],
    dynamic_axes={
        "signals": {0: "batch_size"},
        "logits":  {0: "batch_size"},
    },
    verbose=False,
    dynamo=False,
)

# ── Structural verification ─────────────────────────────────────────────────────
onnx_model = onnx.load(str(OUTPUT_PATH))
onnx.checker.check_model(onnx_model)

# ── Numerical parity check: PyTorch vs ONNX on the same random input ───────────
# onnx.checker only validates graph structure -- it would pass even if the
# export silently produced the wrong computation. Compare actual outputs.
rng = np.random.RandomState(42)
test_input = rng.randn(4, 6, 3840).astype(np.float32)

with torch.no_grad():
    torch_out = model(torch.from_numpy(test_input)).numpy()

session = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"signals": test_input})[0]

max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
if not np.allclose(torch_out, onnx_out, atol=1e-4, rtol=1e-3):
    raise RuntimeError(
        f"ONNX export does not match PyTorch output: max abs diff = {max_abs_diff:.2e}"
    )

# ── Report ───────────────────────────────────────────────────────────────────────
size_bytes = OUTPUT_PATH.stat().st_size
test_f1 = state.get('metrics', {}).get('f1') if isinstance(state, dict) else None

print(f"Checkpoint:        {CHECKPOINT}")
print(f"Checkpoint test F1: {test_f1}")
print(f"ONNX model saved to: {OUTPUT_PATH}")
print(f"  File size: {size_bytes} bytes ({size_bytes / 1024:.1f} KB)")
print(f"  PyTorch vs ONNX max abs diff: {max_abs_diff:.2e} (parity check passed)")
print(f"\nNext: copy this file to your Raspberry Pi and run benchmark_rpi.py")
