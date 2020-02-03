"""
File: parallelized_classifier.py -- Deepfake Detector Creation (Parallelized across GPUs)
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

# Script Parameters

IN_COLAB = False

SOFTMAX = True

BATCH_SIZE = 16
EPOCHS = 5

USE_HHRELU = True
USE_REG= True

USE_NOISE = False
NOISE_TYPE = 'Cauchy'
NOISE_1 = 0
NOISE_2 = 0.01

TRAIN_DIR = "data/train"
VAL_DIR = 'data/test'

# Dynamic Parameters

MODEL_NAMES = ['vgg']
REG_STRENGTHS = [5000]
OUTPUT_BASE = "models/"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from torch.autograd import Variable

# Commented out IPython magic to ensure Python compatibility.
if IN_COLAB: 
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from google.colab import auth
  from oauth2client.client import GoogleCredentials

  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  hhrelu = drive.CreateFile({'id':'1POLUV1n_5pEZcca3fZKfggVoTJz89sjD'})
  hhrelu.GetContentFile('HHReLU.py')
  from google.colab import drive
  drive.mount('/content/gdrive')
#   %cd gdrive/My\ Drive/Fuzzy\ Fakes/code

from HHReLU import HHReLU

def make_model_HHReLU(model, model_name):
  model.relu = HHReLU()
  if 'resnet' in model_name:
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers:
      for i in range(len(layer)):
        layer[i].relu = HHReLU()
  elif 'vgg' in model_name:
    feature_relus = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
    for i in feature_relus:
      model.features[i].relu = HHReLU()
    classifier_relus = [1, 4]
    for i in classifier_relus:
      model.classifier[i].relu = HHReLU()

def lipshitz_regularization(images, model, psi, z=2, train=True, use_softmax=SOFTMAX, num_classes = 2):
    if use_softmax:
      repeated_images = images.repeat(num_classes, 1, 1, 1, 1)
      repeated_output = torch.stack([model(repeated_images[0]).sum(axis=0), model(repeated_images[1]).sum(axis=0)])
      grads = torch.autograd.grad(repeated_output, repeated_images, grad_outputs=torch.eye(num_classes).to(device), create_graph=train)[0]
    else:
      grads = torch.autograd.grad(model(images).sum(), images, create_graph=train)[0]
    return psi*grads.abs().pow(z).mean()

def add_noise(input, noise_type=NOISE_TYPE, noise1=NOISE_1, noise2=NOISE_2):
  if noise_type == 'Gaussian':
    noise = Variable(input.data.new(input.size()).normal_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  elif noise_type == 'Cauchy':
    noise = Variable(input.data.new(input.size()).cauchy_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  elif noise_type == 'Uniform': 
    noise = Variable(input.data.new(input.size()).uniform_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  return input

def train_model(model, criterion, optimizer, scheduler, psi, use_reg=USE_REG, num_epochs=EPOCHS, use_noise=USE_NOISE, use_softmax=SOFTMAX):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                if phase == 'train' and use_noise:
                    inputs = add_noise(inputs)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if use_softmax:
                      _, preds = torch.max(outputs, 1)
                      loss = criterion(outputs, labels)
                    else:
                      preds = torch.where(outputs.squeeze() > 0, torch.ones(len(outputs)).to(device), torch.zeros(len(outputs)).to(device)).long()
                      loss = criterion(outputs, labels.unsqueeze(1).float())

                    if(phase=='train'):
                      if use_softmax:
                        loss = criterion(outputs, labels)
                      else:
                        loss = criterion(outputs, labels.unsqueeze(1).float())
                      
                      if use_reg:
                        inputs.requires_grad = True
                        reg = lipshitz_regularization(inputs, model, psi)
                        inputs.requires_grad = False
                        loss = loss + reg

                      loss.backward()
                      optimizer.step()
                    else:
                      if use_softmax:
                        loss = criterion(outputs, labels)
                      else:
                        loss = criterion(outputs, labels.unsqueeze(1).float())

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

transformation = transforms.Compose([transforms.ToTensor()])

train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transformation)
val_data = datasets.ImageFolder(root=VAL_DIR, transform=transformation)

dataloaders = {} 
dataloaders['train'] = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

dataset_sizes = {}
dataset_sizes['train'] = len(train_data)
dataset_sizes['val'] = len(val_data)

class_names = train_data.classes

for MODEL_NAME in MODEL_NAMES:
  for REG_STRENGTH in REG_STRENGTHS:
    if 'vgg' in MODEL_NAME and REG_STRENGTH==1:
      continue
    OUTPUT_DIR = OUTPUT_BASE + MODEL_NAME + '_reg' + str(REG_STRENGTH)
    print(MODEL_NAME, REG_STRENGTH)

    if 'vgg' in MODEL_NAME:
      model_ft = models.vgg16(pretrained=True)
      num_ftrs = model_ft.classifier[6].in_features
      model_ft.classifier[6] = nn.Linear(num_ftrs, 2 if SOFTMAX else 1)
    elif 'resnet' in MODEL_NAME:
      model_ft = models.resnet18(pretrained=True)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, 2 if SOFTMAX else 1)

    if USE_HHRELU:
      make_model_HHReLU(model_ft, MODEL_NAME)

    if SOFTMAX:
      criterion = nn.CrossEntropyLoss()
    else:
      criterion = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(model_ft)
    if(torch.cuda.device_count() > 1): 
    	model_ft = nn.DataParallel(model_ft)
    model_ft = train_model(model_ft.to(device), criterion, optimizer_ft, exp_lr_scheduler, REG_STRENGTH)

    torch.save(model_ft, OUTPUT_DIR)
    del model_ft