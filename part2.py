import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

class dataset(Dataset):
    def __init__(self, filename, train=True):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())
        self.limits = np.array(self.h5f[str(0)])
        if(train):
            self.transform = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((256,256)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(10),
                                                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                                    transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
        else:
            self.transform = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])

    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, index):
        key = self.keys[index+1]
        data = np.array(self.h5f[key])
        label = 0
        # print(label, key)
        for l in self.limits:
            if(int(key) < l):
                break
            else:
                label += 1
        return self.transform(data), torch.from_numpy(np.array(label)).type(torch.LongTensor)
    
    
    
class CNN(nn.Module):
    def __init__ (self):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            #ip = 3x256x256
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            #op = 4x256x256 
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            #op = 4x128x128
            
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            #op = 16x128x128
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            #op = 16x64x64
            )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(16*64*64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 7),
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

    def train(self, train_data, epochs, learning_rate, batch_size):
        trn_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #loss_fun = nn.CrossEntropyLoss()
        
        losses = [] 
        
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(trn_loader):
                data = torch.autograd.Variable(data)
                optimizer.zero_grad()
                pred = self.forward(data)
                loss = F.cross_entropy(pred, target)
                batch_loss.append(loss.cpu().data.item())
                loss.backward()
                optimizer.step()
            batch_loss = np.array(batch_loss)
            losses.append(batch_loss)
            print('Train epoch: {}, loss: {}'.format(epoch, np.mean(batch_loss)))
        
        return losses   

    def confusion_matrix(self, data):
        data_loader =  DataLoader(data, batch_size=1, num_workers=1, shuffle=False)
        matrix = np.zeros([7,7])
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            pred = self.forward(data)
            pred = pred.detach().numpy().flatten()
            out_class = np.argmax(pred)
            true_class = target.detach().numpy()[0]
            matrix[true_class][out_class] += 1
            total +=1
            if(out_class == true_class):
                correct +=1
        print("Accuracy: ", correct/total)
        return matrix
    

PATH = "~/cs6910/assignment3/"

all_train = False
DATA = dataset("traindata.h5")
train, val = torch.utils.data.dataset.random_split(DATA, [349, 70])


model = CNN()
print(model)

batch_losses = model.train(train, 10, 0.005, 20)
conf1 = model.confusion_matrix(train)
conf2 = model.confusion_matrix(val)
