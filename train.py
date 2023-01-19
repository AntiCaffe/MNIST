import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
mean = trainset.train_data.float().mean(axis=(0,1,2))
std = trainset.train_data.float().std(axis=(0,1,2))
mean = mean /255
std = std/255
""" ##정규화 관련 코드

trainset = dsets.MNIST(root='dataset/',
                       train=True,
                       transform= transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                       download=True)




lr = 0.001
epochs = 20
batchsize = 100



dataset = TensorDataset(trainset.data,trainset.targets)
dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        out = self.layer(x)
        out = out.view(batchsize,-1)
        out = self.fc_layer(out)
        return out

model = CNN().to(device)
cost_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=0.1, patience=1, mode="min")
loss_graph = []

for i in range(1, epochs+1):
    for _,[image,label] in enumerate(dataloader):
        x = image.to(device)
        y = image.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        cost = cost_func(output,y)
        cost.backward()
        optimizer.step()

    scheduler.step(cost)
    print('Epoch : {}, Loss : {}, LR: {}'.format(i,cost.item(),scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
