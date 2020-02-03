"""
File: adv_examples.py -- Adversarial Examples Creation
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

# Script Parameters

MODELS_DIR = 'models/'
MODEL_NAMES = ['resnet_blur3']

ATTACK = 'fgsm'

VAL_DIR = "data/test/"

OUTPUT_DIR = "perturbed_fgsm/"
SOFTMAX = True

# Only use > 1 for carlini wagner
BATCH_SIZE = 1

DEVICE_STR='0'

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
import cv2
from torchvision.utils import save_image
import regex as re
from cw import L2Adversary

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def fgsm(model, loss, eps, softmax=False):
	def attack(img, label):
		output = model(img)
		if softmax: 
			error = loss(output, label)
		else:
			error = loss(output, label.unsqueeze(1).float())
		error.backward()
		perturbed_img = torch.clamp(img + eps*img.grad.data.sign(), 0, 1).detach()
		img.grad.zero_()
		return perturbed_img
	return attack

def ifgsm(model, loss, eps, iters=4, softmax=False):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
			if softmax: 
				error = loss(output, label)
			else:
				error = loss(output, label.unsqueeze(1).float())
			error.backward()
			temp = torch.clamp(perturbed_img + eps*perturbed_img.grad.data.sign(), 0, 1).detach()
			perturbed_img = temp.data
			perturbed_img.requires_grad = True
		return perturbed_img.detach()
	return attack

def bim(model, loss, eps, iters=4, softmax=False):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
			if softmax: 
				error = loss(output, label)
			else:
				error = loss(output, label.unsqueeze(1).float())
			error.backward()
			temp = torch.clamp(perturbed_img + eps*perturbed_img.grad.data, 0, 1).detach()
			#temp = (perturbed_img + eps*perturbed_img.grad.data.sign())
			perturbed_img = temp.data
			perturbed_img.requires_grad = True
		return perturbed_img.detach()
	return attack

def scaled_bim(model, loss, eps, iters=1, softmax=False):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
			if softmax: 
				error = loss(output, label)
			else:
				error = loss(output, label.unsqueeze(1).float())
			error.backward()
			grad = perturbed_img.grad.data 
			grad = grad/torch.max(grad)
			temp = torch.clamp(perturbed_img + eps*grad, 0, 1).detach()
			#temp = (perturbed_img + eps*perturbed_img.grad.data.sign())
			perturbed_img = temp.data
			perturbed_img.requires_grad = True
		return perturbed_img.detach()
	return attack

# Only works with softmax
def carlini(model, device):
  adversary = L2Adversary(targeted=False,
                           confidence=200,
                           c_range=(1e2, 1e4),
                           search_steps=5,
                           max_steps=1000,
                           optimizer_lr=1e-2,
                           init_rand=True)
  def attack(img, label):
    img = img.detach()
    label = label.detach()
    res = adversary(model, img, label, device)
    return res
  return attack

def cw_greedy_round(img, label, model, loss, device):
    img = img.to(device)
    label = label.to(device)
    img.requires_grad = True
    
    output = model(img)
    error = loss(output, label)
    error.backward()
    rounded_img = torch.clamp(img*255.0 + 0.5*img.grad.data.sign(), 0, 255).round().float()/255.0
    
    img.grad.zero_()
    img.requires_grad = False
    
    return rounded_img

def save_batch(model, imgs, labels, paths, criterion, class_names, device, batch_size=BATCH_SIZE):
  for i in range(batch_size):
    label = labels[i]
    orig_path = paths[i]
    img_num = regex.search(orig_path).group(0)
    img_out = model_out + class_names[label] + '/' + img_num + '.jpg'

    if 'cw' in ATTACK:
      rounded_img = cw_greedy_round(imgs[i].unsqueeze(0), label.unsqueeze(0), model, criterion, device)[0]
      img_out_round = model_out_round + class_names[label] + '/' + img_num + '.jpg'

    save_image(imgs[i], img_out)
    if 'cw' in ATTACK:
      save_image(rounded_img, img_out_round)

def generate_adversarial_examples(data_loader, attack, criterion, class_names, device, softmax=False):
	adversarial_examples = list()
	labels = list()
	paths = list()
 
	for img, label, path in tqdm(data_loader):
		img = img.to(device) 
		label = label.to(device)
		
		img.requires_grad = True
		perturbed_img = attack(img, label)
		img.requires_grad = False

		adversarial_examples.append(perturbed_img)
		labels.append(label)
		paths.append(path)
		save_batch(model, perturbed_img, label, path, criterion, class_names, device)

	return adversarial_examples, labels, paths

def test(inputs, labels, model, device):
    running_corrects = 0
    for inputs, labels in tqdm(zip(inputs, labels)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    return (running_corrects.double()/inputs.shape[0])

def test_softmax(inputs, labels, model, device):
    num_correct = 0
    for img, label in tqdm(zip(inputs, labels)):
        img = img.to(device)
        label = label.to(device)
        _, pred = torch.max(model(img), 1)
        if(pred == 1):
            if(label == 1):
                num_correct += 1
        else:
            if(label == 0):
                num_correct += 1

    return num_correct/len(inputs)

def test_softmax_batch(inputs, labels, model, device):
    running_corrects = 0
    for inputs, labels in tqdm(zip(inputs, labels)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    return (running_corrects.double()/inputs.shape[0])

transformation = transforms.Compose([transforms.ToTensor()])

val_data = ImageFolderWithPaths(root=VAL_DIR, transform=transformation)

dataloaders = {} 
dataloaders['val'] = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

dataset_sizes = {}
dataset_sizes['val'] = len(val_data)

class_names = val_data.classes

device = torch.device("cuda:"+DEVICE_STR if torch.cuda.is_available() else "cpu")

if SOFTMAX:
  criterion = nn.CrossEntropyLoss()
else: 
  criterion = nn.BCEWithLogitsLoss()

for mn in MODEL_NAMES:
  print(mn)
  model = torch.load(MODELS_DIR+mn, map_location=device)

  # Create Directories to Output Examples
  model_out = OUTPUT_DIR + mn + '_' + ATTACK + '/'
  os.mkdir(model_out)

  os.mkdir(model_out + class_names[0]+'/')
  os.mkdir(model_out + class_names[1]+'/')
  regex = re.compile(r'\d+')
  if 'cw' in ATTACK:
    model_out_round = OUTPUT_DIR + mn + '_' + ATTACK + '_round/'
    os.mkdir(model_out_round)

    os.mkdir(model_out_round + class_names[0]+'/')
    os.mkdir(model_out_round + class_names[1]+'/')

  # ATTACK: save images after each batch
  if 'fgsm' in ATTACK:
    adv_examples, labels, paths = generate_adversarial_examples(dataloaders["val"], fgsm(model, criterion, 0.02, softmax=SOFTMAX), criterion, class_names, device)
  elif 'cw' in ATTACK:
    adv_examples, labels, paths = generate_adversarial_examples(dataloaders["val"], carlini(model, device), criterion, class_names, device)

  # Evaluate Adversarial Examples
  if BATCH_SIZE > 1:
    print("Acc on adversarial_examples (before saving):", test_softmax_batch(adv_examples, labels, model, device))
  else:
    print("Acc on adversarial_examples (before saving):", test(adv_examples, labels, model) if not SOFTMAX else test_softmax(adv_examples, labels, model, device))