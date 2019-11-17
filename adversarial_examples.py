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
#from cw2 import L2Adversary
from foolbox.attacks import CarliniWagnerL2Attack as cw2
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated


def fgsm(model, loss, eps):
	def attack(img, label):
		output = model(img)
		error = loss(output, label.unsqueeze(1).float())
		error.backward()
		perturbed_img = torch.clamp(img + eps*img.grad.data.sign(), 0, 1).detach()
		#perturbed_img = (img + eps*img.grad.data.sign()).detach()
		img.grad.zero_()
		return perturbed_img
	return attack

def ifgsm(model, loss, eps, iters=4):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
			error = loss(output, label.unsqueeze(1).float())
			error.backward()
			temp = torch.clamp(perturbed_img + eps*perturbed_img.grad.data.sign(), 0, 1).detach()
			#temp = (perturbed_img + eps*perturbed_img.grad.data.sign())
			perturbed_img = temp.data
			perturbed_img.requires_grad = True
		return perturbed_img.detach()
	return attack

def bim(model, loss, eps, iters=4):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
			error = loss(output, label.unsqueeze(1).float())
			error.backward()
			temp = torch.clamp(perturbed_img + eps*perturbed_img.grad.data, 0, 1).detach()
			#temp = (perturbed_img + eps*perturbed_img.grad.data.sign())
			perturbed_img = temp.data
			perturbed_img.requires_grad = True
		return perturbed_img.detach()
	return attack

def scaled_bim(model, loss, eps, iters=4):
	def attack(img, label):
		perturbed_img = img
		perturbed_img.requires_grad = True
		for _ in range(iters):
			output = model(perturbed_img)
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

def cw2(model, confidence=0.0, box=(0., 1.), eps=1e-2, max_iters=1000, abort_early=True, to_numpy=False): 
	'''def attack(img, label):
		return L2Adversary(confidence=confidence, box=box, optimizer_lr=eps, max_steps=max_iters, abort_early=abort_early)(model.cuda(), img.detach().cuda(), label.detach().cuda(), to_numpy=to_numpy).detach()
	return attack'''
	model = PyTorchModel(model, (0, 1), 1)
	def attack(img, label):
		return cw2(model)(img, label)
	return attack

def generate_adversarial_examples(data_loader, attack, device, visualize=False):
	adversarial_examples = list()
	labels = list()
	for img, label in tqdm(data_loader):
		img = img.to(device) 
		label = label.to(device)

		img.requires_grad = True

		perturbed_img = attack(img, label)

		img.requires_grad = False

		if(visualize):
			print("Label: ", label)
			print("Original Predicted: ", model_ft(img))
			print("Perturbed Predicted: ", model_ft(perturbed_img))
		

			imshow(img[0].cpu(), "original")
			imshow(perturbed_img[0].cpu(), "perturbed")
			imshow(torch.clamp(50*(perturbed_img[0] - img[0]), 0, 1).cpu(), "50*difference")
			plt.show()

		adversarial_examples.append(perturbed_img)
		labels.append(label)


	return adversarial_examples, labels

		
def test(inputs, labels, model):
	num_correct = 0
	for img, label in tqdm(zip(inputs, labels)):
		if(model(img).squeeze() > 0):
			if(label == 1):
				num_correct += 1
		else:
			if(label == 0):
				num_correct += 1

	return num_correct/len(inputs)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
#load model
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft.load_state_dict(torch.load("classifier.pt"))
'''

model = torch.load("model_unnormalized.pt")
model_ft = torch.load("model_regularized_unnormalized2.pt")
model.cuda()
model_ft.cuda()

#freeze the model
for p in model_ft.parameters(): 
	p.requires_grad = False

#get data and transform 
transformation = transforms.Compose([Crop(58, 177, 368, 300), 
		transforms.ToTensor(), 
		#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
train_directory = "C:\\Users\\Apurva\\Desktop\\C_Projects\\deepfakes\\train"
test_directory = "C:\\Users\\Apurva\\Desktop\\C_Projects\\deepfakes\\test"
train_dataset = datasets.ImageFolder(train_directory, transform=transformation)
test_dataset = datasets.ImageFolder(test_directory, transform=transformation)
dataloaders = {} 
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(test_dataset)

class_names = train_dataset.classes



criterion = nn.BCEWithLogitsLoss()

#generate_adversarial_examples(dataloaders["val"], scaled_bim(model_ft, criterion, 0.005, iters=100), device)
#adv_examples, labels = generate_adversarial_examples(dataloaders["val"], ifgsm(model, criterion, 0.002), device)
#generate_adversarial_examples(dataloaders["val"], cw2(model), device)
adv_examples, labels = generate_adversarial_examples(dataloaders["val"], fgsm(model, criterion, 0.005), device)


print("Acc on adversarial_examples:", test(adv_examples, labels, model))