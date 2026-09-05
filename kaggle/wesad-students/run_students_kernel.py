"""
Kaggle script kernel: train all student models via KD under nested LOSO.

kernel-metadata.json lists kernel_sources: ["euphora/wesad-teacher"], which
mounts that kernel's /kaggle/working output read-only under /kaggle/input --
no manual "Add Data", no human in the loop. This script copies the 15
teacher checkpoints found there into this run's MODELS_DIR before training.

Not meant to be edited by hand for each run -- kaggle_runner.py rewrites
PINNED_SHA, SMOKE and MODEL_FILTER/MODE_FILTER below before every
`kaggle kernels push`.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

PINNED_SHA = "0940ac16f9b8838620a9dccb1adbb2b1a7f17e5d"
SMOKE = True
MODEL_FILTER = "all"
MODE_FILTER = "both"

REPO_URL = "https://github.com/RiverRover-stack/healthcare_wesad.git"
REPO_DIR = "/kaggle/working/healthcare_wesad"
KERNEL_SLUG = "euphora/wesad-students"


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


def bridge_teacher_checkpoints(models_dir) -> int:
    # MODELS_DIR comes from src/config.py, resolved from WESAD_OUTPUT_DIR --
    # must be the absolute path, not a path relative to the cloned repo.
    found = glob.glob('/kaggle/input/**/teacher_loso_S*.pt', recursive=True)
    assert found, (
        'No teacher checkpoints found under /kaggle/input -- '
        'kernel_sources should mount euphora/wesad-teacher there.'
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in found:
        shutil.copy(f, models_dir)
    copied = sorted(models_dir.glob('teacher_loso_S*.pt'))
    print(f'Copied {len(copied)} teacher checkpoints into {models_dir}')
    if not SMOKE:
        assert len(copied) == 15, f'Expected 15 teacher checkpoints, found {len(copied)}'
    return len(copied)


def main() -> None:
    os.environ['WESAD_DATA_DIR'] = detect_data_root()
    os.environ['WESAD_OUTPUT_DIR'] = '/kaggle/working/outputs'
    print('WESAD_DATA_DIR  :', os.environ['WESAD_DATA_DIR'])
    print('WESAD_OUTPUT_DIR:', os.environ['WESAD_OUTPUT_DIR'])

    clone_and_checkout()
    os.chdir(REPO_DIR)

    install_requirements()

    sys.path.insert(0, 'src')
    from config import MODELS_DIR, create_directories
    create_directories()
    n_teacher_ckpts = bridge_teacher_checkpoints(MODELS_DIR)

    import torch
    print('CUDA available:', torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print('Device:', gpu_name or 'CPU (!)')
    assert torch.cuda.is_available(), 'GPU not enabled for this kernel -- check kernel-metadata.json.'

    cmd = [sys.executable, 'train_students.py']
    if SMOKE:
        cmd += ['--smoke', '--model', 'MicroCNN', '--mode', 'both', '--skip-efficiency']
    else:
        if MODEL_FILTER != 'all':
            cmd += ['--model', MODEL_FILTER]
        if MODE_FILTER != 'both':
            cmd += ['--mode', MODE_FILTER]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    assert result.returncode == 0, f'train_students.py failed with exit code {result.returncode}'

    import json
    kaggle_meta = {
        'kernel_slug': KERNEL_SLUG, 'pinned_sha': PINNED_SHA, 'smoke': SMOKE,
        'model_filter': MODEL_FILTER, 'mode_filter': MODE_FILTER,
        'gpu_name': gpu_name, 'n_teacher_checkpoints_bridged': n_teacher_ckpts,
    }
    with open('/kaggle/working/outputs/kaggle_meta.json', 'w', encoding='utf-8') as f:
        json.dump(kaggle_meta, f, indent=2)


if __name__ == '__main__':
    main()
