"""
Kaggle Driver: Push, Poll, and Download Kaggle Script Kernels

Drives the three script kernels under kaggle/wesad-{teacher,students,ablation}/:
pins each kernel script to a git SHA before pushing (so a later push to
GitHub can't silently change what a re-run executes), pushes it, polls
`kaggle kernels status` until it reaches a terminal state, and downloads
the kernel's /kaggle/working output.

Usage:
    python kaggle_runner.py teacher --smoke
    python kaggle_runner.py students
    python kaggle_runner.py students --model MicroCNN --mode standalone
    python kaggle_runner.py ablation --sweep temperature

Never runs two kernels concurrently -- the caller is expected to wait for
one phase's kernel to finish (this script blocks until it does) before
starting the next.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# kaggle CLI prints filenames/progress with characters that crash on Windows'
# default cp1252 console encoding (UnicodeEncodeError -> non-zero exit,
# indistinguishable from a real download failure). Force UTF-8 for every
# kaggle subprocess call.
_KAGGLE_ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

KERNEL_DIRS = {
    'main': Path('kaggle/wesad-main'),
    'teacher': Path('kaggle/wesad-teacher'),
    'students': Path('kaggle/wesad-students'),
    'ablation': Path('kaggle/wesad-ablation'),
}
KERNEL_SCRIPTS = {
    'main': 'run_main_kernel.py',
    'teacher': 'run_teacher_kernel.py',
    'students': 'run_students_kernel.py',
    'ablation': 'run_ablation_kernel.py',
}
KERNEL_SLUGS = {
    # Kaggle derives the slug from the kernel *title*, not the "id" field in
    # kernel-metadata.json, when they'd otherwise disagree -- title "WESAD
    # Teacher (script)" became "wesad-teacher-script", not "wesad-teacher".
    # kernel-metadata.json's "id" fields were updated to match this reality.
    'main': 'euphora/wesad-main-script',
    'teacher': 'euphora/wesad-teacher-script',
    'students': 'euphora/wesad-students-script',
    'ablation': 'euphora/wesad-ablation-script',
}
POLL_INTERVAL_SEC = 60
MAX_WAIT_SEC = 12 * 3600  # Kaggle's own per-session ceiling
TERMINAL_STATUSES = ('complete', 'error', 'cancelAcknowledged', 'cancelled')


def parse_args():
    parser = argparse.ArgumentParser(description='Push/poll/download a WESAD Kaggle kernel')
    parser.add_argument('phase', choices=list(KERNEL_DIRS.keys()))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--sha', default=None, help='Git SHA to pin (default: current local HEAD)')
    parser.add_argument('--model', default=None, help='students: --model filter; ablation: student model (default MicroCNN)')
    parser.add_argument('--mode', default=None, help='students: --mode filter (standalone/distilled/both)')
    parser.add_argument('--sweep', default=None, help='ablation: --sweep (temperature/alpha/both)')
    parser.add_argument('--no-wait', action='store_true', help='Push and return immediately, do not poll')
    return parser.parse_args()


def current_git_sha() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def rewrite_constant(script: Path, name: str, value: str, quoted: bool) -> None:
    """Rewrite `NAME = ...` at module level in a kernel script."""
    text = script.read_text(encoding='utf-8')
    new_val = f'"{value}"' if quoted else str(value)
    pattern = rf'^{name} = .*$'
    replacement = f'{name} = {new_val}'
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    assert n == 1, f'{name} line not found in {script}'
    script.write_text(new_text, encoding='utf-8')


def configure_kernel(phase: str, args) -> Path:
    kernel_dir = KERNEL_DIRS[phase]
    script = kernel_dir / KERNEL_SCRIPTS[phase]
    sha = args.sha or current_git_sha()

    rewrite_constant(script, 'PINNED_SHA', sha, quoted=True)
    if phase != 'main':
        rewrite_constant(script, 'SMOKE', args.smoke, quoted=False)

    if phase == 'students':
        rewrite_constant(script, 'MODEL_FILTER', args.model or 'all', quoted=True)
        rewrite_constant(script, 'MODE_FILTER', args.mode or 'both', quoted=True)
    elif phase == 'ablation':
        rewrite_constant(script, 'MODEL', args.model or 'MicroCNN', quoted=True)
        rewrite_constant(script, 'SWEEP', args.sweep or 'both', quoted=True)

    print(f'Configured {script} -> PINNED_SHA={sha[:8]} SMOKE={args.smoke}')
    return kernel_dir


def push(kernel_dir: Path) -> None:
    subprocess.run(['kaggle', 'kernels', 'push', '-p', str(kernel_dir)], check=True, env=_KAGGLE_ENV)


def poll(slug: str, max_wait: int = MAX_WAIT_SEC) -> str:
    waited = 0
    consecutive_failures = 0
    while waited < max_wait:
        result = subprocess.run(['kaggle', 'kernels', 'status', slug],
                                capture_output=True, text=True, env=_KAGGLE_ENV)
        if result.returncode != 0:
            consecutive_failures += 1
            print(f'  [{waited // 60:>4}m] status check failed (attempt {consecutive_failures}): '
                  f'{result.stderr.strip()}')
            if consecutive_failures >= 5:
                raise RuntimeError(
                    f'kaggle kernels status {slug} failed {consecutive_failures} times in a row -- '
                    f'likely a wrong slug (check kernel-metadata.json "id" vs the actual pushed URL) '
                    f'rather than a transient API issue.'
                )
            time.sleep(POLL_INTERVAL_SEC)
            waited += POLL_INTERVAL_SEC
            continue

        consecutive_failures = 0
        status = result.stdout.strip()
        print(f'  [{waited // 60:>4}m] {status}')
        if any(s.lower() in status.lower() for s in TERMINAL_STATUSES):
            return status
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC
    raise TimeoutError(f'{slug} did not reach a terminal state within {max_wait}s')


def download(slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out_dir), '--force'],
                   check=True, env=_KAGGLE_ENV, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    kernel_dir = configure_kernel(args.phase, args)
    slug = KERNEL_SLUGS[args.phase]

    print(f'Pushing {kernel_dir} as {slug}...')
    push(kernel_dir)

    if args.no_wait:
        print('--no-wait set; not polling. Check status with:')
        print(f'  kaggle kernels status {slug}')
        return

    status = poll(slug)
    print(f'Final status: {status}')

    out_dir = Path('outputs_kaggle') / args.phase
    download(slug, out_dir)
    print(f'Downloaded output -> {out_dir}')

    push_record = {'phase': args.phase, 'slug': slug, 'final_status': status,
                   'downloaded_to': str(out_dir)}
    (out_dir / 'kaggle_push_info.json').write_text(json.dumps(push_record, indent=2))

    if 'error' in status.lower():
        sys.exit(1)


if __name__ == '__main__':
    main()
