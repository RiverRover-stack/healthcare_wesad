import numpy as np
import pytest

from conftest import fake_index, DEFAULT_SUBJECTS
from training.loso import make_fold_split


def test_folds_are_disjoint_and_complete():
    subject_per_window, label_per_window = fake_index()
    all_subjects = sorted(set(subject_per_window))
    n_windows = len(subject_per_window)

    for fold_idx, test_subject in enumerate(all_subjects):
        split = make_fold_split(fold_idx, test_subject, all_subjects,
                                 subject_per_window, label_per_window)
        assert split is not None

        train_subjects = set(subject_per_window[split.train_idx])
        val_subjects = set(split.val_subjects)
        test_subjects = {test_subject}

        assert len(split.val_subjects) == 2
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)
        assert train_subjects | val_subjects | test_subjects == set(all_subjects)

        partition = sorted(split.train_idx + split.val_idx + split.test_idx)
        assert partition == list(range(n_windows))


def test_rotation_matches_documented_formula():
    subject_per_window, label_per_window = fake_index()
    all_subjects = sorted(set(subject_per_window))

    for fold_idx, test_subject in enumerate(all_subjects):
        train_subjects = sorted(s for s in all_subjects if s != test_subject)
        n = len(train_subjects)
        expected = [train_subjects[fold_idx % n], train_subjects[(fold_idx + 1) % n]]

        split = make_fold_split(fold_idx, test_subject, all_subjects,
                                 subject_per_window, label_per_window)
        assert split.val_subjects == expected


def test_single_class_test_subject_is_skipped():
    subject_per_window, label_per_window = fake_index(single_class_subjects={'S2': 0})
    all_subjects = sorted(set(subject_per_window))

    split = make_fold_split(0, 'S2', all_subjects, subject_per_window, label_per_window)
    assert split is None


def test_rotation_advances_past_single_class_val_pair():
    # sorted(all_subjects) = S10 S11 S13 S14 S15 S16 S17 S2 S3 S4 S5 S6 S7 S8 S9
    # test_subject='S9' -> train_subjects[0:2] = ('S10', 'S11'), forced single-class
    subject_per_window, label_per_window = fake_index(
        single_class_subjects={'S10': 0, 'S11': 0}
    )
    all_subjects = sorted(set(subject_per_window))

    split = make_fold_split(0, 'S9', all_subjects, subject_per_window, label_per_window)
    assert split is not None
    assert split.val_subjects == ['S11', 'S13']


def test_sampler_and_weights_are_inner_train_only():
    subject_per_window, label_per_window = fake_index()
    all_subjects = sorted(set(subject_per_window))

    for fold_idx, test_subject in enumerate(all_subjects):
        split = make_fold_split(fold_idx, test_subject, all_subjects,
                                 subject_per_window, label_per_window)
        assert split is not None
        train_subjects_used = set(subject_per_window[split.train_idx])
        assert test_subject not in train_subjects_used
        assert train_subjects_used.isdisjoint(set(split.val_subjects))
