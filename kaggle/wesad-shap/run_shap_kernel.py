"""
Kaggle script kernel: SHAP channel-attribution analysis over all 15 LOSO folds.

kernel-metadata.json lists kernel_sources: ["euphora/wesad-students-script"],
which mounts that kernel's /kaggle/working output read-only under
/kaggle/input. The students kernel's own MODELS_DIR already has both the 15
bridged teacher checkpoints and the 90 student checkpoints, so mounting it
alone is enough -- no need to also mount the teacher kernel separately.

Not meant to be edited by hand for each run -- kaggle_runner.py rewrites
PINNED_SHA below before every `kaggle kernels push`.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

PINNED_SHA = "0000000000000000000000000000000000000000"  # rewritten by kaggle_runner.py

REPO_URL = "https://github.com/RiverRover-stack/healthcare_wesad.git"
REPO_DIR = "/kaggle/working/healthcare_wesad"
KERNEL_SLUG = "euphora/wesad-shap-analysis-script"


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
    """Skip torch/onnx/onnxruntime -- Kaggle's preinstalled torch is already
    matched to whatever GPU this session got (see the teacher/students/ablation
    kernels for the full story: a pip-installed torch build dropped support
    for the P100 this account was assigned once)."""
    lines = Path('requirements.txt').read_text(encoding='utf-8').splitlines()
    skip = ('torch', 'onnx', 'onnxruntime')
    filtered = [l for l in lines if not any(l.strip().lower().startswith(p) for p in skip)]
    Path('requirements_kaggle.txt').write_text('\n'.join(filtered), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements_kaggle.txt'],
                   check=True)


def bridge_checkpoints(models_dir) -> int:
    found = glob.glob('/kaggle/input/**/*.pt', recursive=True)
    assert found, (
        'No checkpoints found under /kaggle/input -- '
        'kernel_sources should mount euphora/wesad-students-script there.'
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in found:
        shutil.copy(f, models_dir)
    copied = list(models_dir.glob('*.pt'))
    n_teacher = len([p for p in copied if p.name.startswith('teacher_loso_')])
    n_student = len([p for p in copied if p.name.startswith('MicroCNN_distilled_loso_')])
    print(f'Copied {len(copied)} checkpoints into {models_dir} '
          f'({n_teacher} teacher, {n_student} MicroCNN-distilled)')
    assert n_teacher == 15, f'Expected 15 teacher checkpoints, found {n_teacher}'
    assert n_student == 15, f'Expected 15 MicroCNN-distilled checkpoints, found {n_student}'
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
    n_ckpts = bridge_checkpoints(MODELS_DIR)

    import torch
    print('CUDA available:', torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print('Device:', gpu_name or 'CPU (!)')

    cmd = [sys.executable, 'shap_analysis.py']
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    assert result.returncode == 0, f'shap_analysis.py failed with exit code {result.returncode}'

    import json
    kaggle_meta = {
        'kernel_slug': KERNEL_SLUG, 'pinned_sha': PINNED_SHA,
        'gpu_name': gpu_name, 'n_checkpoints_bridged': n_ckpts,
    }
    with open('/kaggle/working/outputs/kaggle_meta.json', 'w', encoding='utf-8') as f:
        json.dump(kaggle_meta, f, indent=2)


if __name__ == '__main__':
    main()
