import numpy as np
import torch

from conftest import DEFAULT_SUBJECTS
from segmentation.window_data import WindowedData
from training.trainer import train_teacher_loso
from training.loso import SMOKE_FOLDS

SIGNAL_CHANNELS = ['chest_ecg', 'chest_eda', 'chest_emg', 'chest_resp', 'chest_temp', 'chest_acc']
TARGET_LENGTH = 3840  # 64 Hz * 60 s -- makes the 700->64 Hz resample a near-free identity


def _make_windowed_data(n_per_subject=20, seed=0):
    rng = np.random.RandomState(seed)
    windowed = {}
    for sid in DEFAULT_SUBJECTS:
        windows = {ch: rng.randn(n_per_subject, TARGET_LENGTH).astype(np.float32)
                   for ch in SIGNAL_CHANNELS}
        half = n_per_subject // 2
        labels = np.array([0] * half + [1] * (n_per_subject - half))
        rng.shuffle(labels)
        windowed[sid] = WindowedData(
            subject_id=sid, windows=windows, labels=labels,
            num_windows=n_per_subject,
            num_baseline=int(np.sum(labels == 0)),
            num_stress=int(np.sum(labels == 1)),
        )
    return windowed


def test_smoke_teacher_training(capsys):
    from config import MODELS_DIR

    windowed = _make_windowed_data()
    results = train_teacher_loso(windowed, smoke=True)
    assert results

    captured = capsys.readouterr().out
    assert 'val=[' in captured
    assert 'best_ep=' in captured
    assert 'valF1=' in captured and 'testF1=' in captured

    for sid in sorted(DEFAULT_SUBJECTS)[:SMOKE_FOLDS]:
        ckpt_path = MODELS_DIR / f"teacher_loso_{sid}.pt"
        assert ckpt_path.exists()

        ckpt = torch.load(ckpt_path, weights_only=True)
        assert set(ckpt.keys()) == {
            'model_state', 'subject', 'mode', 'val_subjects', 'metrics',
            'val_metrics', 'final_epoch_metrics', 'best_epoch', 'n_epochs', 'seed',
        }
        assert sid not in ckpt['val_subjects']
        assert 1 <= ckpt['best_epoch'] <= ckpt['n_epochs']
