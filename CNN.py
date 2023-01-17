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

trainset = dsets.MNIST(root='dataset/',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)
testset = dsets.MNIST(root='dataset/',
                      train=False,
                      transform=transforms.ToTensor(),
                      download=True)

print(trainset.data.shape)
print(trainset.targets.shape)
print(testset.data.shape)
print(testset.targets.shape)
mean = trainset.train_data.float().mean(axis=(0,1,2))
std = trainset.train_data.float().std(axis=(0,1,2))


mean = mean /255
std = std/255

lr = 0.0001
epochs = 20
batchsize = 50



dataset = TensorDataset(trainset.data,trainset.targets)
dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = True)
testloader = DataLoader(testset,batch_size= batchsize,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.ReLU = nn.ReLU();
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.Conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.Flatten = nn.Flatten()
        self.FC1 = nn.Linear(400, 120, bias=True)
        self.FC2 = nn.Linear(120, 84, bias=True)
        self.FC3 = nn.Linear(84, 10, bias=True)
        self.Softmax = nn.Softmax(dim=1)

        self.layer = nn.Sequential(
            self.Conv1,
            self.ReLU,
            self.MaxPool,
            self.Conv2,
            self.ReLU,
            self.MaxPool,
            self.Flatten,
            self.FC1,
            self.ReLU,
            self.FC2,
            self.ReLU,
            self.FC3,
            self.Softmax
        )
    def forward(self,x):
        out = self.layer(x)
        return out

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.999))
loss_graph = []
for k in range(epochs +1):
  for i, sample in enumerate(dataloader):
    (x,y) = sample
    x = x.unsqueeze(dim = 1)
    z = model(x.float().to(device))
    cost = f.cross_entropy(z,y.to(device)).to(device)
    optimizer.zero_grad() ##미분 계수 0으로 만듬
    cost.backward()
    optimizer.step()

    if i == 49:
      print("{0}/{1}, {3}/{2}, {4}" .format(k,epochs,int(trainset.data.size(0)/batchsize),i,cost.item()))
      loss_graph.append(cost.item())
