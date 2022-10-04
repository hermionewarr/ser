
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ser.info import get_commit
from ser.train import train as run_train
from ser.infer import inference
from ser.constants import PROJECT_ROOT, Parameters, load_params
from ser.transforms import normalize, flip

import typer

main = typer.Typer()

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")


### TRAINING ENTRYPOINT ###
@main.command() #to run: ser model-setup --name ect
def train(name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."),
        include_flip: bool = typer.Option(
        False, "-f", "--flip", help="flip transform to be applied during training."), #eg 'experiment2/2022_09_30-04:57:56_PM'
        epochs: int = typer.Option(
            2, "-e", "--epochs", help="number of epochs to train the model for."
        ),
        batch_size: int = typer.Option(
            1000, "-b", "--batch_size", help="batch size."
        ),
        learning_rate: float = typer.Option(
            0.01, "-lr", "--learning_rate", help="learning rate."
        )
        ):

    #set up
    commit = get_commit(PROJECT_ROOT)
    params = Parameters(name, epochs, batch_size, learning_rate, commit, date)
    print("\nData directory: ", params.DATA_DIR, "\n")
    print("\nSave directory: ", params.SAVE_DIR, "\n")
    print("\nResults directory: ", params.RESULTS_DIR, "\n")
    params.make_save_dir()
    params.make_results_dir()
    
    ts = [normalize]
    if include_flip: 
        ts = [normalize, flip]
        print("Flip transform selected.")

    #train model
    run_train(params, ts)
    
    return params

### INFERENCE ENTRYPOINT ###
@main.command()
def infer(name: str = typer.Option(
        ..., "-n", "--name", help="Name of run model saved under."), #eg 'experiment2/2022_09_30-04:57:56_PM'
        include_flip: bool = typer.Option(
        False, "-f", "--flip", help="flip transform to be applied during training."), #eg 'experiment2/2022_09_30-04:57:56_PM'
         
        ):
    
    MODEL_DIR = PROJECT_ROOT / 'runs' / name / 'model'
    print('Model dir: ', MODEL_DIR)

    #commit = get_commit(PROJECT_ROOT)    
    
    #params_test = Parameters(name, epochs, batch_size, learning_rate, commit, DATA_DIR)
    if include_flip: 
        ts = [normalize, flip]
        print("Flip transform selected.")

    inference(MODEL_DIR, ts)
    
    pass
