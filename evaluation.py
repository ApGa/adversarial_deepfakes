"""
File: evaluation.py -- Model Evaluation Script
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

# Script Parameters

DIR_NAMES = ['perturbed_cw/vgg_reg1000_cw']

MODELS_DIR = 'models/'
MODEL_NAMES = ['vgg_blur']

OUTPUT_DIR = 'results.csv'

SOFTMAX = True

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
from sklearn import metrics
import scipy
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

def input_gradient(images, model, z=2, use_softmax=SOFTMAX, num_classes = 2):
    if use_softmax:
      repeated_images = images.repeat(num_classes, 1, 1, 1, 1)
      repeated_output = torch.stack([model(repeated_images[0]).sum(axis=0), model(repeated_images[1]).sum(axis=0)])
      grads = torch.autograd.grad(repeated_output, repeated_images, grad_outputs=torch.eye(num_classes).to(device), create_graph=False)[0]
    else:
      grads = torch.autograd.grad(model(images).sum(), images, create_graph=False)[0]
    return grads.abs().pow(z).mean().cpu().detach().numpy()

def evaluate(model, data, device, use_softmax=SOFTMAX):
  labels = []
  scores = []
  preds = []
  grads = []
  losses = []

  for img, label in tqdm(data):      
      img = img.to(device) 
      output = model(img)

      # Calculate Gradient With Respect to Input
      img.requires_grad = True
      grad = input_gradient(img, model, use_softmax)
      img.requires_grad = False
      grads.append(grad)

      # Calculate Loss 
      if use_softmax:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label.to(device))
      else:
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, label.unsqueeze(1).float())
      losses.append(loss.squeeze().cpu().detach().numpy())

      # Calculate prediction
      if use_softmax:
        score, pred = torch.max(output, 1)
        try:
          score = np.exp(output[:, 1].cpu().detach().numpy())/(output.flatten().exp().sum().cpu().detach().numpy())
        except:
          print('Error applying softmax to model output. Setting output to 0.5.')
          score = 0.5
        pred = pred.squeeze().cpu().detach().numpy()
      else:
        score = scipy.special.expit(output.squeeze().cpu().detach().numpy())
        pred = 1 if (score >= 0.5) else 0

      labels.append(label)
      scores.append(score)
      preds.append(pred)
      del img, score, pred, loss, label, output
    
  return labels, scores, preds, grads, losses

results = pd.DataFrame(columns=['Model_Name', 'Dataset', 'AUC', 'Accuracy', 'Precision_fake', 'Precision_real', 'Recall_fake', 'Recall_real', 'Input_Gradient', 'Loss'])

for dn in DIR_NAMES: 

  transformation = transforms.Compose([transforms.ToTensor()])
  imgfolder = datasets.ImageFolder(dn, transform=transformation)
  data = torch.utils.data.DataLoader(imgfolder, batch_size=1, shuffle=True)
  class_names = imgfolder.classes

  for mn in MODEL_NAMES:
    print('Evaluating model:', mn, 'on dataset:', dn)

    model = torch.load(MODELS_DIR+mn, map_location=device)
    model.to(device)

    labels, scores, preds, grads, losses = evaluate(model, data, device)

    res = {}
    res['Dataset'] = dn
    res['Model_Name'] = mn
    res['Accuracy'] = round(metrics.accuracy_score(labels, preds)*100, 3)
    try:
      res['AUC'] = round(metrics.roc_auc_score(labels, scores)*100, 3)
    except:
      print('AUC is undefined. Setting to NaN.')
      res['AUC'] = np.nan
    res['Precision_'+class_names[0]] = round(metrics.precision_score(labels, preds, pos_label=0)*100, 3)
    res['Precision_'+class_names[1]] = round(metrics.precision_score(labels, preds, pos_label=1)*100, 3)
    res['Recall_'+class_names[0]] = round(metrics.recall_score(labels, preds, pos_label=0)*100, 3)
    res['Recall_'+class_names[1]] = round(metrics.recall_score(labels, preds, pos_label=1)*100, 3)
    res['Input_Gradient'] = round(np.mean(grads), 5)
    res['Loss'] = round(np.mean(losses), 5)

    results = results.append(res, ignore_index=True)

results.to_csv(OUTPUT_DIR, index=False)

