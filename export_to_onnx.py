"""
Export MicroCNN to ONNX for Raspberry Pi deployment.

Run this on your main machine AFTER training students:
    python export_to_onnx.py

Output: outputs/models/micro_cnn.onnx
"""

import torch
import os
from src.models.student import MicroCNN

# ── Load your trained MicroCNN checkpoint ─────────────────────────────────────
# Uses S2 fold (trained on S3-S17, representative model)
CHECKPOINT = "outputs/models/MicroCNN_distilled_loso_S2.pt"
OUTPUT_PATH = "outputs/models/micro_cnn.onnx"

os.makedirs("outputs/models", exist_ok=True)

model = MicroCNN(in_channels=6, num_classes=2)
state = torch.load(CHECKPOINT, map_location="cpu")

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
    OUTPUT_PATH,
    export_params=True,
    opset_version=11,           # RPi onnxruntime supports opset 11
    input_names=["signals"],
    output_names=["logits"],
    dynamic_axes={
        "signals": {0: "batch_size"},
        "logits":  {0: "batch_size"},
    },
    verbose=False,
)

# ── Verify export ──────────────────────────────────────────────────────────────
import onnx
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"✓ ONNX model saved to: {OUTPUT_PATH}")
print(f"  File size: {size_kb:.1f} KB")
print(f"\nNext: copy this file to your Raspberry Pi and run benchmark_rpi.py")
