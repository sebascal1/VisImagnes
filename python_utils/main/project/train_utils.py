#Original author jpcano1, used under authortization, content belongs to him#
#Dice function implementation sebascal1#

import copy

from torch.nn import functional as F
import torch

from tqdm.auto import tqdm

import numpy as np

from . import general as gen

from collections import OrderedDict

def get_lr(opt):
    """
    Obtain the learning rate of the optimizer
    :param opt: The optimizer of the model
    :return: The learning rate of the optimizer
    """
    # Loop through the opt params
    for param_group in opt.param_groups:
        return param_group['lr']

def jaccard(y_pred, y_true, dim=(2, 3), eps=1e-5):
    """
    Intersection over Union metric
    :param y_pred: The predictions of the model
    :param y_true: The true labels of the data
    :param dim: The axis where we calculate the operations
    :param eps: The tolerance
    :return: The loss calculated and the IoU metric
    """
    # Intersection
    inter = torch.sum(y_true * y_pred, dim=dim)
    # Union
    union = torch.sum(y_pred, dim=dim) + torch.sum(y_true, dim=dim)
    union -= inter
    # The whole metric
    IoU = ((inter + eps) / (union + eps)).mean()
    loss = 1 - IoU
    return loss, IoU

def loss_func(y_pred, y_true, metric):
    """
    The loss function calculator
    :param y_pred: The predictions of the model
    :param y_true: The true labels of the data
    :param metric: The metric of the model
    :return: The loss calculated and the metric
    """
    # We take the binary cross-entropy
    # for binary classification
    bce = F.binary_cross_entropy(y_pred, y_true, 
                                 reduction="mean")
    # Loss and metric
    loss, acc = metric(y_pred, y_true)
    # Sum the binary cross-entropy
    loss += bce
    return loss, acc

def batch_loss(criterion, y_pred, y_true, metric, opt=None):
    """
    The loss per batch in the dataset
    :param criterion: The loss functions
    :param y_pred: The predictions of the model
    :param y_true: The true labels of the data
    :param metric: The metric of the model
    :param opt: The optimizer
    :return: The loss calculated and the metric
    """
    # Calculate loss and metric
    loss, acc = criterion(y_pred, y_true, metric)

    # Backpropagation method if train mode
    if opt is not None:
        # Clean the gradients of the optimizer
        opt.zero_grad()
        # Apply the backpropagation algorithm
        loss.backward()
        # Apply the gradients over the neural net
        opt.step()

    # Return the numbers of the loss and the metric
    return loss.item(), acc.item()

def epoch_loss(model, criterion, metric, dataloader, device,
               sanity_check=False, opt=None, epoch=None):
    """
    The loss per epoch
    :param model: The model to be trained
    :param criterion: The loss function
    :param metric: The metric function
    :param dataloader: The dataloader
    :param device: The hardware accelerator device
    :param sanity_check: The sanity check flag
    :param opt: The optimizer of the model
    :return: The loss and the metric per epoch
    """
    epoch_loss = 0.
    epoch_acc = 0.

    len_data = len(dataloader)

    bar = tqdm(dataloader)

    if epoch is not None:
        bar.set_description(f"Epoch {epoch}")

    if opt is not None:
        loss_key = "train_loss"
        acc_key = "train_acc"
    else:
        loss_key = "val_loss"
        acc_key = "val_acc"

    status = OrderedDict()
    # Loop over each data batch
    for X_batch, y_batch in bar:

        # Allocate the data in the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Make the predictions
        y_pred = model(X_batch)

        # Calculate the batch loss
        b_loss, b_acc = batch_loss(criterion, y_pred, 
                                   y_batch, metric, opt)
        epoch_loss += b_loss
        epoch_acc += b_acc

        # Update bar status
        status[loss_key] = b_loss
        status[acc_key] = b_acc * 100.

        bar.set_postfix(status)
        if sanity_check:
            break
    
    bar.close()
    # Calculate the mean
    loss = epoch_loss / float(len_data)
    acc = epoch_acc / float(len_data)

    return loss, acc

def evaluate(model, criterion, dataloader, device, 
             sanity_check, metric=jaccard):
    """
    Method that evaluates the model on a dataloader
    :param model: The model to e evaluated
    :param criterion: The loss function
    :param dataloader: The dataloader
    :param device: The hardware accelerator device
    :param sanity_check: The sanity check flag
    :param metric: The metric function
    :return: The loss calculated and the metric
    """
    # Deactivate all layers
    model.eval()

    # Deactivate the PyTorch AutoGrad
    with torch.no_grad():
        # Calculate the validation loss and accuracy
        loss, acc = epoch_loss(model, criterion, metric,
                               dataloader, device, sanity_check)
    return loss, acc

def train(model, epochs, criterion, opt, train_dl, val_dl, 
          sanity_check, lr_scheduler, weights_dir, device,
          metric=jaccard, **kwargs):
    """
    The training loop
    :param model: The model to be trained and evaluated
    :param epochs: The number of epochs per training
    :param criterion: The loss function
    :param opt: The optimizer
    :param train_dl: The train dataloader
    :param val_dl: The validation data loader
    :param sanity_check: The sanity check flag
    :param lr_scheduler: The scheduler for the learning rate
    :param weights_dir: The directory of weights
    :param device: The hardware accelerator device
    :param metric: The metric function
    :param kwargs: Function Keyword arguments
    :return: The best model trained and the history
    """
    # Loss history dictionary
    loss_history = {
        "train": [],
        "val": []
    }

    # Accuracy history dictionary
    acc_history = {
        "train": [],
        "val": []
    }

    # Best parameters
    best_model = copy.deepcopy(model.state_dict())
    best_loss = kwargs.get("best_loss") or float("inf")
    best_acc = kwargs.get("best_acc") or 0

    # Loop through the epochs
    for epoch in tqdm(range(epochs)):
        current_lr = get_lr(opt)

        # Activate all layers
        model.train()
        # Calculate the train loss and accuracy
        train_loss, train_acc = epoch_loss(model, criterion, metric,
                                           train_dl, device, sanity_check,
                                           opt, epoch + 1)
        # Append to the dictionaries
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)

        val_loss, val_acc = evaluate(model, criterion, val_dl,
                                     device, sanity_check)

        # Append to the dictionaries
        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_acc)

        # Conditional to have the best loss and best accuracy
        if val_loss < best_loss or val_acc > best_acc:
            best_loss = val_loss
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

            # Save best model
            torch.save(model.state_dict(), weights_dir)
            print("Copied best model weights!")

        # We call the scheduler to analyse the val loss
        lr_scheduler.step(val_loss)

        # If the scheduler changed the learning rate
        # Take the best model weights.
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model)

        print(f"Train Loss: {train_loss:.6f}, Accuracy: {100 * train_acc:.2f}")
        print(f"Val loss: {val_loss:.6f}, Accuracy: {100 * val_acc:.2f}")
        print("-"*50)

        if sanity_check:
            break

    # Load best model and return
    model.load_state_dict(best_model)
    return model, loss_history, acc_history

class SimpleGenerator:
    def __init__(self, path2data):
        self.data_dir = gen.read_listdir(path2data[0])
        self.labels_dir = gen.read_listdir(path2data[1])

    def __len__(self):
        return len(self.data_dir)

    def size(self):
        return len(self)

    def __getitem__(self, idx):
        path2img = self.data_dir[idx]
        path2lab = self.labels_dir[idx]

        X = np.load(path2img)
        y = np.load(path2lab)

        return X, y
