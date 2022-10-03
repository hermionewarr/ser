#all data transforms defined here
from torchvision import transforms

# torch transforms
def transform(type: str = 'basic', *stages):
    if type == 'basic':
        ts = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
       ts = transforms.Compose(
            [transforms.ToTensor(),  *(stage() for stage in stages)]
            #add some augmentations
        )
    return ts

def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return transforms.Normalize((0.5,), (0.5,))


def flip():
    """
    Flip a tensor both vertically and horizontally
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]
    )