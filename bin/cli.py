
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ser.info import get_commit
from ser.train import train
from ser.infer import inference
from ser.constants import PROJECT_ROOT, DATA_DIR, Parameters
from ser.transforms import normalize, flip

import typer

main = typer.Typer()

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

'''PROJECT_ROOT = Path(__file__).parent.parent
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
'''

### TRAINING ENTRYPOINT ###
@main.command() #to run: ser model-setup --name ect
def model_setup(name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."),
        epochs: int = typer.Option(
            2, "-e", "--epochs", help="number of epochs to train the model for."
        ),
        batch_size: int = typer.Option(
            1000, "-b", "--batch_size", help="batch size."
        ),
        learning_rate: float = typer.Option(
            0.01, "-lr", "--learning_rate", help="learning rate."
        ),
        DATA_DIR = DATA_DIR
        ):

    print("\nData directory: ", DATA_DIR, "\n")
    
    SAVE_DIR = PROJECT_ROOT / 'runs' / name / date / 'model'
    RESULTS_DIR = PROJECT_ROOT / 'runs' / name / date / 'results'
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    commit = get_commit(PROJECT_ROOT)

    params = Parameters(name, epochs, batch_size, learning_rate, commit, DATA_DIR, SAVE_DIR, RESULTS_DIR)

    #train model
    train(params)
    
    return params

### INFERENCE ENTRYPOINT ###
@main.command()
def infer(name: str = typer.Option(
        ..., "-n", "--name", help="Name of run model saved under."), #eg 'experiment2/2022_09_30-04:57:56_PM'
        transforms: bool = typer.Option(
        False, "-f", "--flip", help="list of transforms to be applied during training."), #eg 'experiment2/2022_09_30-04:57:56_PM'
         
        ):
    
    MODEL_DIR = PROJECT_ROOT / 'runs' / name / 'model'
    print('Model dir: ', MODEL_DIR)

    #commit = get_commit(PROJECT_ROOT)    
    
    #params_test = Parameters(name, epochs, batch_size, learning_rate, commit, DATA_DIR)
    if transforms: 
        ts = [normalize, flip]
        print("Flip transform selected.")

    inference(MODEL_DIR, ts)
    
    pass
