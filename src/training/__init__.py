"""Training package: nested-LOSO fold logic (loso.py), teacher (trainer.py)
and student/KD (models.distillation) trainers. No re-exports here -- import
submodules directly to avoid a circular import between trainer.py,
models.distillation, and training.loso."""
