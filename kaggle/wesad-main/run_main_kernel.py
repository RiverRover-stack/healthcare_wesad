"""
Kaggle script kernel: run main.py (baselines, LogReg, RF, dataset_stats.json).

CPU-only (kernel-metadata.json: enable_gpu: false) -- main.py never touches
torch. This kernel exists purely to sidestep a local memory constraint on
the development machine; the classical pipeline itself is unchanged.

Not meant to be edited by hand for each run -- kaggle_runner.py rewrites
PINNED_SHA below before every `kaggle kernels push`.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

PINNED_SHA = "0000000000000000000000000000000000000000"  # rewritten by kaggle_runner.py

REPO_URL = "https://github.com/RiverRover-stack/healthcare_wesad.git"
REPO_DIR = "/kaggle/working/healthcare_wesad"
KERNEL_SLUG = "euphora/wesad-main-script"


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
    """Skip torch/onnx/onnxruntime -- main.py doesn't import torch at all."""
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

    cmd = [sys.executable, 'main.py']
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    assert result.returncode == 0, f'main.py failed with exit code {result.returncode}'

    import json
    kaggle_meta = {'kernel_slug': KERNEL_SLUG, 'pinned_sha': PINNED_SHA}
    with open('/kaggle/working/outputs/kaggle_meta.json', 'w', encoding='utf-8') as f:
        json.dump(kaggle_meta, f, indent=2)


if __name__ == '__main__':
    main()
