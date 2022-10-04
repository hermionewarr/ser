import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import copy 
import pandas as pd
import visdom
from ser.display import VisdomLinePlotter
from ser.constants import save_params

global vis
vis = VisdomLinePlotter(env_name='SER Plots')

def train_loop(model_params, params):
    train_losses = []
    val_accuracy = []
    val_losses = []
    val_best = 0
    #loss_plot = vis.line(X = torch.zeros((1)).cpu(), Y = torch.zeros((1)).cpu(), opts=dict(showlegend=True, title='train loss', xlabel = 'batch*epoch', ylabel = 'loss', legend=['train loss']))
    
    for epoch in range(1,(params.epochs +1)):
        _train_batch(model_params, epoch, train_losses)
        val_best, best_model_dict = _val_batch(model_params, epoch, val_accuracy, val_losses, val_best)
    
    train_losses = pd.DataFrame({'train_loss' : train_losses})
    val_accuracy = pd.DataFrame({'val_acc':val_accuracy, 'val_loss': val_losses})
    acc_dict = {'train_losses': train_losses, 'val_accuracy' : val_accuracy}
    return acc_dict, best_model_dict, val_best

def _train_batch(model_params, epoch, train_losses):
    for batch, (images, labels) in enumerate(model_params.dataloaders['training_dataloader']):
        images, labels = images.to(model_params.device), labels.to(model_params.device)
        model_params.model.train()
        model_params.optimizer.zero_grad()
        output = model_params.model(images)
        #pred = output.argmax(dim=1, keepdim=True)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model_params.optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {batch}/{len(model_params.dataloaders['training_dataloader'])} "
            f"| Loss: {loss.item():.4f}"
        )
        train_losses.append(loss.item()) #maybe add batch and epoch info here
        #vis_update(batch, epoch, loss, vis, loss_plot) 
        vis.plot('loss', 'train', 'Train Loss', epoch*batch, loss.item())

    return

@torch.no_grad()
def _val_batch(model_params, epoch, val_accuracy, val_losses, val_best):
    val_loss = 0
    correct = 0
    best_epoch = 0
    best_model_state = model_params.model.state_dict()
    for images, labels in model_params.dataloaders['validation_dataloader']:
        images, labels = images.to(model_params.device), labels.to(model_params.device)
        model_params.model.eval()
        output = model_params.model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(model_params.dataloaders['validation_dataloader'].dataset)
    val_acc = correct / len(model_params.dataloaders['validation_dataloader'].dataset)
    if val_acc > val_best:
        val_best = val_acc
        best_epoch = epoch
        best_model_state = copy.deepcopy(model_params.model.state_dict())#save this at end. 
    print(
        f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
    )
    val_accuracy.append(val_acc)
    val_losses.append(val_loss)
    print(
        f"|* Best Epoch: {best_epoch} | Best Accuracy: {val_best} *|"
        )  
    return val_best, best_model_state

#plotting loss as you go
def vis_update(batch, epoch, loss):
    vis.line(X=torch.ones((1,1)).cpu()*batch*epoch, Y =torch.ones((1,1)).cpu()*loss.item(), win = loss_plot, update='append')
    return

def save_outputs(params, model_dict, acc_dict):
    ### SAVING THE RESULTS ###
    torch.save(model_dict, params.SAVE_DIR / 'model_dict.pt')

    save_params(params, params.SAVE_DIR)

    acc_dict['train_losses'].to_csv(params.RESULTS_DIR / 'train_loss.csv', index=False)
    acc_dict['val_accuracy'].to_csv(params.RESULTS_DIR / 'val_accuracy_loss.csv')

    return