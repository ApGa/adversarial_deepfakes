from __future__ import print_function, division

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
from crop import Crop
from HHReLU import HHReLU

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def make_model_HHReLU(model):
    model.relu = HHReLU()
    model.layer1[0].relu = HHReLU()
    model.layer1[1].relu = HHReLU()
    model.layer2[0].relu = HHReLU()
    model.layer2[1].relu = HHReLU()
    model.layer3[0].relu = HHReLU()
    model.layer3[1].relu = HHReLU()
    model.layer4[0].relu = HHReLU()
    model.layer4[1].relu = HHReLU()

def lipshitz_regularization(images, model, z=2, train=True, psi=1000):
    grads = torch.autograd.grad(model(images).sum(), images, create_graph=train)[0]
    return psi*grads.abs().pow(z).mean()



#train script 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
                inputs = inputs.to(device)
                labels = labels.to(device)


                #imshow(inputs[0].detach().cpu())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    preds = torch.where(outputs.squeeze() > 0, torch.ones(len(outputs)).cuda(), torch.zeros(len(outputs)).cuda()).long()
                    if(phase=='train'):
                        inputs.requires_grad = True
                        reg = lipshitz_regularization(inputs, model)
                        inputs.requires_grad = False
                        loss = criterion(outputs, labels.unsqueeze(1).float())
                        #print(loss, reg)
                        loss = loss + reg

                        loss.backward()
                        optimizer.step()
                    else:
                        loss = criterion(outputs, labels.unsqueeze(1).float())
                

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#get data and transform 
transformation = transforms.Compose([Crop(58, 177, 368, 300), 
		transforms.ToTensor()
		#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
train_directory = "C:\\Users\\Apurva\\Desktop\\C_Projects\\deepfakes\\train"
test_directory = "C:\\Users\\Apurva\\Desktop\\C_Projects\\deepfakes\\test"
train_dataset = datasets.ImageFolder(train_directory, transform=transformation)
test_dataset = datasets.ImageFolder(test_directory, transform=transformation)
dataloaders = {} 
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(test_dataset)

class_names = train_dataset.classes

#visualize


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft.load_state_dict(torch.load("classifier0.pt"))

make_model_HHReLU(model_ft)

print(model_ft)

criterion = nn.BCEWithLogitsLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = train_model(model_ft.cuda(), criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 10)

#torch.save(model_ft.state_dict(), "classifier.pt")
torch.save(model_ft, "model_regularized_unnormalized2.pt")



