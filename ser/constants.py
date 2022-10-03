from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
print('Root dir: ', PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"

@dataclass
class Parameters():
     name: str 
     epochs: int
     batch_size: int
     learning_rate: float
     commit: str
     DATA_DIR: str
     SAVE_DIR: str
     RESULTS_DIR: str