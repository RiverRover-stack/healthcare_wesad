"""
Kaggle script kernel: train the WESAD teacher CNN under nested LOSO.

Not meant to be edited by hand for each run -- kaggle_runner.py rewrites
PINNED_SHA and SMOKE below before every `kaggle kernels push`, so what a
kernel version actually executed is always pinned to one commit, not
whatever `main` happens to be when it runs.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

PINNED_SHA = "0940ac16f9b8838620a9dccb1adbb2b1a7f17e5d"
SMOKE = True

REPO_URL = "https://github.com/RiverRover-stack/healthcare_wesad.git"
REPO_DIR = "/kaggle/working/healthcare_wesad"
KERNEL_SLUG = "euphora/wesad-teacher"


def detect_data_root() -> str:
    candidates = glob.glob('/kaggle/input/**/S2/S2.pkl', recursive=True)
    if not candidates:
        print('No match. Top-level /kaggle/input contents:',
              os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else '(missing)')
    assert candidates, (
        "WESAD dataset not found under /kaggle/input. "
        "Attach orvile/wesad-wearable-stress-affect-detection-dataset."
    )
    return os.path.dirname(os.path.dirname(candidates[0]))


def clone_and_checkout() -> None:
    if not os.path.isdir(REPO_DIR):
        subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)
    subprocess.run(['git', '-C', REPO_DIR, 'checkout', PINNED_SHA], check=True)
    sha = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    print(f'Checked out {sha}')


def install_requirements() -> None:
    """
    Install everything in requirements.txt except torch/onnx/onnxruntime.
    Kaggle's preinstalled torch is already the build matched to whatever GPU
    this session got; pip installing the pinned version from requirements.txt
    (needed for local reproduction, where nothing is preinstalled) overwrote
    it with a newer build that dropped support for the P100's compute
    capability (sm_60), so every conv layer errored with "no kernel image is
    available for execution on the device" the first time a smoke run landed
    on a P100 instead of a T4.
    """
    lines = Path('requirements.txt').read_text(encoding='utf-8').splitlines()
    skip = ('torch', 'onnx', 'onnxruntime')
    filtered = [l for l in lines if not any(l.strip().lower().startswith(p) for p in skip)]
    Path('requirements_kaggle.txt').write_text('\n'.join(filtered), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements_kaggle.txt'],
                   check=True)


def main() -> None:
    os.environ['WESAD_DATA_DIR'] = detect_data_root()
    os.environ['WESAD_OUTPUT_DIR'] = '/kaggle/working/outputs'
    print('WESAD_DATA_DIR  :', os.environ['WESAD_DATA_DIR'])
    print('WESAD_OUTPUT_DIR:', os.environ['WESAD_OUTPUT_DIR'])

    clone_and_checkout()
    os.chdir(REPO_DIR)

    install_requirements()

    import torch
    print('CUDA available:', torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print('Device:', gpu_name or 'CPU (!)')
    assert torch.cuda.is_available(), 'GPU not enabled for this kernel -- check kernel-metadata.json.'

    cmd = [sys.executable, 'train_teacher.py']
    if SMOKE:
        cmd.append('--smoke')
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    assert result.returncode == 0, f'train_teacher.py failed with exit code {result.returncode}'

    ckpts = sorted(glob.glob('/kaggle/working/outputs/models/teacher_loso_S*.pt'))
    print(f'{len(ckpts)} teacher checkpoints found')
    if not SMOKE:
        assert len(ckpts) == 15, f'Expected 15 checkpoints, found {len(ckpts)}'

    import json
    kaggle_meta = {
        'kernel_slug': KERNEL_SLUG, 'pinned_sha': PINNED_SHA, 'smoke': SMOKE,
        'gpu_name': gpu_name, 'n_checkpoints': len(ckpts),
    }
    with open('/kaggle/working/outputs/kaggle_meta.json', 'w', encoding='utf-8') as f:
        json.dump(kaggle_meta, f, indent=2)


if __name__ == '__main__':
    main()
