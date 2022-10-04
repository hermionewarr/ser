from crypt import methods
from datetime import datetime
from pathlib import Path
import torch
from dataclasses import asdict, dataclass
import json 

PROJECT_ROOT = Path(__file__).parent.parent
print('Root dir: ', PROJECT_ROOT)
#DATA_DIR = PROJECT_ROOT / "data"

@dataclass
class model_parameters():
    model: object
    optimizer: str
    device: torch.device
    dataloaders: dict
    hyperparams: dataclass

@dataclass
class Parameters():
     name: str 
     epochs: int
     batch_size: int
     learning_rate: float
     commit: str
     date: str

     @property
     def DATA_DIR(self):
          return PROJECT_ROOT / 'data'

     @property
     def SAVE_DIR(self):
          print(self.name)
          return PROJECT_ROOT / 'runs' / str(self.name) / str(self.date) / 'model'

     @property 
     def RESULTS_DIR(self):
          return PROJECT_ROOT / 'runs' / str(self.name) / str(self.date) / 'results'
     
     def make_save_dir(self):
          self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
          return

     def make_results_dir(self):
          self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
          return


def factory(data):
     return {k:(str(v) if k=='date' else v) for k,v in data}

def save_params(params, save_path = None):
     if save_path == None:
          save_path = params.SAVE_DIR
     with open(save_path / 'parameters.json', 'w') as file:
        #params.SAVE_DIR
        json.dump(asdict(params, dict_factory=factory), file, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        file.close()

def load_params(model_dir):
	with open(model_dir / 'parameters.json', "r") as f:
		data = json.load(f)
		for key, i in data.items():
			print(key ,': ', i)
		return Parameters(**data)	