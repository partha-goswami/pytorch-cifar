import torch
import torchvision
from torchvision import datasets, transforms
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,OneCycleLR
import seaborn as sns
import matplotlib.pyplot as plt

def get_device():
  '''
  This method returns the device in use.
  If cuda(gpu) is available it would return that, otherwise it would return cpu.
  '''
  use_cuda = torch.cuda.is_available()
  return torch.device("cuda" if use_cuda else "cpu")

def get_config_values():
  '''
  Returns the config data used
  '''

  dict_config_values = {}
  dict_config_values['dropout_rate'] = 0.05
  dict_config_values['batch_size'] = 256
  dict_config_values['no_of_workers'] = 2
  dict_config_values['pin_memory'] = True
  dict_config_values['learning_rate'] = 0.001
  dict_config_values['epochs'] = 20
  dict_config_values['L1_factor'] = 0
  dict_config_values['L2_factor'] = 0.0001

  
  dict_config_values['albumentation'] = {
      'randomCrop_Size': 32,
      'randomCrop_Padding': 4,

      'coarseDropout_max_holes': 1,
      'coarseDropout_min_holes': 1,
      'coarseDropout_max_height': 16,
      'coarseDropout_max_width': 16,
      'coarseDropout_min_height': 16,
      'coarseDropout_min_width': 16,
      'coarseDropout_cutout_probability': 0.5
  }
  return dict_config_values

def get_cifar10_stats():
  '''
  we are calculating mean and standard deviation for each channel of the input data. We would use this in cutout fillvalue in albumentation transformation
  '''
  train_transform = transforms.Compose([transforms.ToTensor()])
  train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  return train_set.data.mean(axis=(0,1,2))/255, train_set.data.std(axis=(0,1,2))/255

def apply_albumentation(config_dict):
  '''
  Kept separate method to apply albumentation. As it returns the transformation applied in terms of dictionary.
  Our next dataset and dataloaders would use the transformation by calling this method.
  '''
  cifar10_mean,cifar10_std = get_cifar10_stats()

  train_transforms = albumentations.Compose([albumentations.RandomCrop(config_dict['albumentation']['randomCrop_Size'],
                                                                      padding=config_dict['albumentation']['randomCrop_Padding']),
                                  albumentations.CoarseDropout(max_holes=config_dict['albumentation']['coarseDropout_max_holes'],
                                                               min_holes =config_dict['albumentation']['coarseDropout_min_holes'], 
                                                               max_height=config_dict['albumentation']['coarseDropout_max_height'], 
                                                               max_width=config_dict['albumentation']['coarseDropout_max_width'], 
                                  p=config_dict['albumentation']['coarseDropout_cutout_probability'],fill_value=tuple([x * 255.0 for x in cifar10_mean]),
                                  min_height=config_dict['albumentation']['coarseDropout_min_height'], min_width=config_dict['albumentation']['coarseDropout_min_width']),
                                  albumentations.Normalize(mean=cifar10_mean, std=cifar10_std,always_apply=True),
                                  ToTensorV2()
                                ])

  test_transforms = albumentations.Compose([albumentations.Normalize(mean=cifar10_mean, std=cifar10_std, always_apply=True),
                                 ToTensorV2()])
  return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]


def get_data_loaders(config_dict):
  '''
  This method applies albumentation transforms and returns the train and test dataloaders
  : param config_dict: dictionary of config values
  '''
  train_transforms, test_transforms = apply_albumentation(config_dict)  
  

  trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)  
        
  testset  = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

  train_loader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=config_dict['batch_size'], 
                                                  shuffle=True,
                                                  num_workers=config_dict['no_of_workers'], 
                                                  pin_memory=config_dict['pin_memory'])
  test_loader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=config_dict['batch_size'], 
                                                  shuffle=True,
                                                  num_workers=config_dict['no_of_workers'], 
                                                  pin_memory=config_dict['pin_memory'])
  return train_loader, test_loader

def get_optimizer(config_dict, model):
  '''
  This method returns the optimizer
  :param config_dict: config dictionary
  :param model: model
  '''
  return optim.Adam(model.parameters(), lr=config_dict['learning_rate'], weight_decay=config_dict['L2_factor'])

def get_scheduler(train_loader, config_dict, model):
  '''
  This method returns the scheduler
  :param train_loader: train loader
  :param config_dict: config dictionary
  :param model: model
  '''
  optimizer = get_optimizer(config_dict, model)
  return OneCycleLR(optimizer, max_lr=config_dict['learning_rate'],epochs=config_dict['epochs'],steps_per_epoch=len(train_loader)), optimizer

