from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class MesoNet(nn.Module):
	def __init__(self):
		super(MesoNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
		self.bn1 = nn.BatchNorm2d(num_features=8)
		self.pool1 = nn.MaxPool2d(kernel_size=2)

		self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5)
		self.bn2 = nn.BatchNorm2d(num_features=8)
		self.pool2 = nn.MaxPool2d(kernel_size=2)

		self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
		self.bn3 = nn.BatchNorm2d(num_features=16)
		self.pool3 = nn.MaxPool2d(kernel_size=2)

		self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
		self.bn4 = nn.BatchNorm2d(num_features=16)
		self.pool4 = nn.MaxPool2d(kernel_size=4)

		self.do5 = nn.Dropout(p=0.5)
		self.dense5 = nn.Linear(in_features=8*16*9, out_features=16)
		self.lr5 = nn.LeakyReLU()

		self.do6 = nn.Dropout(p=0.5)
		self.dense6 = nn.Linear(in_features=16, out_features=1)

	def forward(self, x):
		x = self.pool1(self.bn1(F.relu(self.conv1(x))))
		x = self.pool2(self.bn2(F.relu(self.conv2(x))))
		x = self.pool3(self.bn3(F.relu(self.conv3(x))))
		x = self.pool4(self.bn4(F.relu(self.conv4(x))))

		x = self.do5(x)
		print(x.size())
		x = self.dense5(x)
		x = self.lr5(x)
		x = self.dense6(self.do6(x))

		return x
