#input data specified here
#MINST DATA
from torch.utils.data import DataLoader
from torchvision import datasets

from ser.transforms import transform

#dataset
def setup_dataset(params, ts, train_val_test = 'train'):
    if train_val_test == 'train':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=True, transform=transform('train',*ts))
    elif train_val_test == 'val':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=False, transform=transform('val',*ts))
    elif train_val_test == 'test':
        dataset = datasets.MNIST(root=params.DATA_DIR, download=True, train=False, transform=transform('test',*ts))
    return dataset

#dataloaders
def dataloaders(params, batch_size, ts, train_val_test = 'train'):
    dataloader = DataLoader(
        setup_dataset(params, ts, train_val_test),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return dataloader

""" 
# dataloaders
def dataloaders(params, batch_size, ts):
    training_dataloader = DataLoader(
        setup_dataset(params, ts, train_val_test = 'train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        setup_dataset(params, ts, train_val_test = 'val'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    
    return training_dataloader, validation_dataloader
 """    
def test_dataloader(DATA_DIR, batch_size, ts, shuffle = False):
    #same as val for now
    test_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform('test',*ts)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_dataloader