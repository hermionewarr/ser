#e2e tests on small dataset

from ser.train_utils import train_loop, save_outputs
from ser.data import dataloaders
from ser.constants import Parameters
from ser.transforms import normalize
import torch
from torch import optim
from ser.CNN_model import Net
from ser.constants import model_parameters
import visdom
from torch.utils.data import DataLoader
from torchvision import datasets
import pytest
from ser.transforms import transform


#dataset
def setup_dataset(params, ts, split = None, train_val_test = 'train'):
    if train_val_test == 'train':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=True, transform=transform('train',*ts))
    elif train_val_test == 'val':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=False, transform=transform('val',*ts))
    elif train_val_test == 'test':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=False, transform=transform('test',*ts))
    
    if split is not None:
        dataset = torch.utils.data.Subset(dataset, (0,10))

    return dataset

#dataloaders
def dataloaders(params, batch_size, ts, split, train_val_test = 'train'):
    dataloader = DataLoader(
        setup_dataset(params, ts, split,train_val_test),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return dataloader

def test_e2e():
    transforms = [normalize]
    params = Parameters("test", 1, 10, 1.0, "abcdefg", 'today')
    params.make_results_dir()
    params.make_save_dir()

    print(f"Running experiment {params.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    training_dataloader = dataloaders(params, params.batch_size, transforms, split =10, train_val_test = 'train')
    validation_dataloader = dataloaders(params, params.batch_size, transforms, split =10 ,train_val_test = 'val')
    
    model_params = model_parameters(model, optimizer, device, {'training_dataloader': training_dataloader, 'validation_dataloader': validation_dataloader}, params)
    
    ### TRAINING LOOP ###
    acc_dict, best_model_dict, val_best = train_loop(model_params, params)
    save_outputs(params, best_model_dict, acc_dict)


