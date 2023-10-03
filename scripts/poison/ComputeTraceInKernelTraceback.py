#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from einops import rearrange
import os
import numpy as np
from scipy.stats import kendalltau, spearmanr


# In[2]:


with open('../DATA/poison/cifar_cifar2_res.p','rb') as f:
    RES = pickle.load(f)


# In[3]:


class Model(torch.nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        
        self.conv2d = torch.nn.Conv2d(3, 32, (3,3), padding=(1,1),)
        self.batch_normalization = torch.nn.BatchNorm2d(32,momentum=0.01,eps=1e-3)
        self.conv2d_1 = torch.nn.Conv2d(32, 32, (3,3), padding=(1,1))
        self.batch_normalization_1 = torch.nn.BatchNorm2d(32,momentum=0.01,eps=1e-3)
        self.max_pooling2d = torch.nn.MaxPool2d((2,2))
        
        self.conv2d_2 = torch.nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.batch_normalization_2 = torch.nn.BatchNorm2d(64,momentum=0.01,eps=1e-3)
        self.conv2d_3 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1))
        self.batch_normalization_3 = torch.nn.BatchNorm2d(64,momentum=0.01,eps=1e-3)
        self.max_pooling2d_1 = torch.nn.MaxPool2d((2,2))
        
        self.conv2d_4 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1))
        self.batch_normalization_4 = torch.nn.BatchNorm2d(128,momentum=0.01,eps=1e-3)
        self.conv2d_5 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.batch_normalization_5 = torch.nn.BatchNorm2d(128,momentum=0.01,eps=1e-3)
        self.max_pooling2d_2 = torch.nn.MaxPool2d((2,2))

        self.flatten = torch.nn.Flatten()
        self.max_pooling1d = torch.nn.MaxPool1d((4))
        self.dropout = torch.nn.Dropout(0.2)
        
        self.dense = torch.nn.Linear(512,512,)
        self.batch_normalization_6 = torch.nn.BatchNorm1d(512,momentum=0.01,eps=1e-3)
        
        self.dense_1 = torch.nn.Linear(512,512,)
        self.batch_normalization_7 = torch.nn.BatchNorm1d(512,momentum=0.01,eps=1e-3)
        
        self.dense_2 = torch.nn.Linear(512,10,)
    
    def forward(self,x):
        x = self.conv2d(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization(x)
        x = self.conv2d_1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_1(x)
        x = self.max_pooling2d(x)
        
        x = self.conv2d_2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_2(x)
        x = self.conv2d_3(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_3(x)
        x = self.max_pooling2d_1(x)
        
        x = self.conv2d_4(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_4(x)
        x = self.conv2d_5(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_5(x)
        x = self.max_pooling2d_2(x)
        
        
        x = self.flatten(x)
        x = self.max_pooling1d(x)
        x = self.dropout(x)
        
        x = self.dense(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_6(x)
        x = self.dense_1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_normalization_7(x)
        x = self.dense_2(x)
        
        return x


# In[4]:


model = Model()
model.to('cuda')
torch.manual_seed(1234)

optim = torch.optim.SGD(model.parameters(),1e-2,momentum=0.9,nesterov=True)
loss = torch.nn.CrossEntropyLoss()


# In[5]:


RES.keys()

train_x = rearrange(torch.tensor(RES['injected_X'],dtype=torch.float32),'b h w c -> b c h w')
train_y = torch.tensor(RES['injected_Y'],dtype=torch.float32)

test_x = rearrange(torch.tensor(RES['injected_X_test'],dtype=torch.float32),'b h w c -> b c h w')
test_y = torch.tensor(RES['injected_Y_test'],dtype=torch.float32)

test_x_og = rearrange(torch.tensor(RES['X_test'],dtype=torch.float32),'b h w c -> b c h w')
test_y_og = torch.tensor(RES['Y_test'],dtype=torch.float32)



train_loader = DataLoader(TensorDataset(train_x,train_y),batch_size=128,shuffle=True)
test_loader = DataLoader(TensorDataset(test_x,test_y),batch_size=1,shuffle=False)
test_loader_og = DataLoader(TensorDataset(test_x_og,test_y_og),batch_size=1,shuffle=False)


# In[6]:


all_X = torch.cat([train_x,test_x,test_x_og]).to('cuda')


# In[7]:


model.load_state_dict(torch.load('../MODELS/poison/poisoned_CNN.pt'))


total_kernel = 0
model.eval()
for name,param in model.named_parameters():
    TraceIn_Component = torch.zeros((70_000,70_000),device='cpu')
    
    for i in tqdm(range(7)):
        X1 = []
        for k in range(i*10_000,(i+1)*10_000):
            model.zero_grad()
            Y = model.forward(all_X[k:k+1])
            Loss = loss(Y.squeeze(),all_Y[k:k+1].squeeze())
            Loss.backward()
            
            X1.append(param.grad.flatten())
        
        X1 = torch.stack(X1) # B x P
        for j in range(7):
            X2 = []
            for k in range(j*10_000,(j+1)*10_000):
                model.zero_grad()
                Y = model.forward(all_X[k:k+1])
                Loss = loss(Y.squeeze(),all_Y[k:k+1].squeeze())
                Loss.backward()
                
                X2.append(param.grad.flatten())
            
            X2 = torch.stack(X2) # B x P
            
            TraceIn_Component[i*10_000:(i+1)*10_000,j*10_000:(j+1)*10_000] = torch.matmul(X1,X2.T).cpu()
    
    total_kernel+= TraceIn_Component
    break


# In[13]:


np.save('../KERNELS/poison/TraceIn.npy',total_kernel.detach().cpu().numpy())
