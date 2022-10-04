#model training code defined here
from dataclasses import dataclass
import dataclasses
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
import pandas as pd
import visdom

from ser.CNN_model import Net
from ser.data import dataloaders
from ser.train_utils import save_outputs, _train_batch, _val_batch, train_loop
from ser.constants import model_parameters

import typer
main = typer.Typer()
#in terminal run: visdom
#vis = visdom.Visdom()


### TRAINING FUNTION ###
def train(params, ts):
    print(f"Running experiment {params.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    #training_dataloader, validation_dataloader = dataloaders(params.DATA_DIR, params.batch_size, ts)
    training_dataloader = dataloaders(params, params.batch_size, ts, train_val_test = 'train')
    validation_dataloader = dataloaders(params, params.batch_size, ts ,train_val_test = 'val')
    
    model_params = model_parameters(model, optimizer, device, {'training_dataloader': training_dataloader, 'validation_dataloader': validation_dataloader}, params)
    
    ### TRAINING LOOP ###
    acc_dict, best_model_dict, val_best = train_loop(model_params, params)
    save_outputs(params, best_model_dict, acc_dict)
   
    return 

