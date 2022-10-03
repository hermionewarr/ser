from ser.transforms import transform, normalize, flip
from ser.data import test_dataloader
from ser.display import display_num
import numpy as np 
import torch

def _select_testimage(label):
    ts = [normalize, flip]
    dataloader1 = test_dataloader(1, transform('test',*ts))
    dataloader2 = test_dataloader(1, transform('test',*ts[0])) 
    images, labels = next(iter(dataloader1))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader1))
    
    return images

def test_transform():
    ts = [flip]
    t = transform('test', *ts)
    im = np.array([[1.0,2.0,2.0], [1.0,2.0,3.0], [1.0,2.0,3.0]])
    expected = torch.tensor([[2.,2.,1.],[3.,2.,1.],[3.,2.,1.]])
    expected = torch.tensor([[[3.,2.,1.],[3.,2.,1.],[2.,2.,1.]]]) 

    assert torch.equal(t(im), expected)
