#e2e tests on small dataset

from ser.train_utils import train_loop, save_outputs
from ser.data import dataloaders, test_dataloader
from ser.constants import Parameters
from ser.transforms import normalize
import torch
from torch import optim
from ser.CNN_model import Net
from ser.constants import model_parameters
import visdom


def test_e2e():
    vis1 = visdom.Visdom()
    transforms = [normalize]
    params = Parameters("test", 1, 1, 1.0, "abcdefg", 'today')
    train_dataloader = test_dataloader(params.DATA_DIR, 10, transforms, shuffle =True)
    #validation_dataloader = train_dataloader
    #images, labels = next(iter(dataloader)) #get one batch
    
    print(f"Running experiment {params.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    #training_dataloader, validation_dataloader = dataloaders(params.DATA_DIR, params.batch_size, ts=[normalize])
    
    model_params = model_parameters(model, optimizer, device, {'training_dataloader': iter(train_dataloader), 'validation_dataloader': next(iter(train_dataloader))}, params)
    
    ### TRAINING LOOP ###
    acc_dict, best_model_dict, val_best = train_loop(model_params, params, vis1)
    #save_outputs(params, best_model_dict, acc_dict)


