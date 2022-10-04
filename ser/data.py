#input data specified here
#MINST DATA
from torch.utils.data import DataLoader
from torchvision import datasets

from ser.transforms import transform

# dataloaders
def dataloaders(DATA_DIR, batch_size, ts):
    training_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=transform('train',*ts)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    
    return training_dataloader, validation_dataloader
    
def test_dataloader(DATA_DIR, batch_size, ts, shuffle = False):
    #same as val for now
    test_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform('test',*ts)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_dataloader