from .trainer import train_teacher_loso

# Import via sys.path to avoid relative-import issues when src/ is on sys.path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.distillation import train_student_kd_loso
