import torch
import torch.nn
import torchvision
from torchvision import datasets, transforms
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import pickle


def train(model, device, train_loader, optimizer, epoch, train_acc, train_loss, l1_factor, scheduler, criterion,
          grad_clip=None):
    '''
    This method is responsible for model training
    :param model: model
    :param device: cuda (gpu) or cpu
    :param train_loader: train loader
    :param optimizer: Optimizer, for example, Adam, or SGD
    :param epoch: epoch, the number of times we are seeing the entire training data
    :param train_acc: training accuracy
    :param train_loss: training loss
    :param l1_factor: L1 Factor
    :param scheduler: scheduler
    :param criterion: criterion
    :param grad_clip: gradient clipping value
    '''

    model.train()
    pbar = tqdm(train_loader)
    correct, processed = 0, 0

    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)

        if l1_factor > 0:
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + l1_factor * l1

        train_loss.append(loss.data.cpu().numpy().item())
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, test_acc, test_losses, criterion):
    '''
    This method is responsible for model testing
    :param model: model
    :param device: device, cuda (gpu), or cpu
    :param test_loader: test loader
    :param test_acc: test accuracy
    :param test_losses: test losses
    :param criterion: criterion
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


def print_model_summary(model, device):
    '''
      This method returns the model summary.
      :param model: model
      :param device: device
    '''
    cifar_model = model.to(device)
    return summary(cifar_model, input_size=(3, 32, 32))


def experiment(model):
    '''
    The method performs the experiment as per our configuration.
    To preserve gpu usage, we are exiting while we reach the target validation accuracy.
    '''

    config_dict = utils.get_config_values()
    train_loader, test_loader = utils.get_data_loaders(config_dict)
    # print( train_loader, test_loader)

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    device = utils.get_device()
    model = model.to(device)
    scheduler, optimizer = utils.get_scheduler(train_loader, config_dict, model)

    for epoch in range(1, config_dict['epochs'] + 1):
        print(f'Epoch {epoch}:')
        train(model, device, train_loader, optimizer, epoch, train_accuracy, train_losses, config_dict['L1_factor'],
              scheduler, nn.CrossEntropyLoss())

        test(model, device, test_loader, test_accuracy, test_losses, nn.CrossEntropyLoss())

    with open('data.pickle', 'wb') as f:
        pickle.dump((train_accuracy, train_losses, test_accuracy, test_losses), f)

    return (train_accuracy, train_losses, test_accuracy, test_losses)
