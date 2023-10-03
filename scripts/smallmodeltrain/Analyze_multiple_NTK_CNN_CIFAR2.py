#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')

import numpy as np
import torch
import random

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch import load
from torch.nn import functional as F
from torch import autograd

from torchvision import datasets

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import time

import sys
from pathlib import Path

from tqdm import tqdm

import os

import gc

import torchntk.autograd as ezntk
import torchvision.models as models
from tqdm import tqdm

from einops import rearrange

device='cuda'


# In[3]:


train_data = datasets.CIFAR10(
    root = '../../DATA/',
    train = True,                          
    download = True,            
)


test_data = datasets.CIFAR10(
    root = '../../DATA/', 
    train = False, 
    download=True,
)

train_x = torch.tensor(train_data.data)
test_x = torch.tensor(test_data.data)

train_y = torch.tensor(train_data.targets)
test_y = torch.tensor(test_data.targets)

train_mask = torch.logical_or(train_y==0,train_y==1)
test_mask = torch.logical_or(test_y==0,test_y==1)

train_x = train_x[train_mask]
train_y = train_y[train_mask]

test_x = test_x[test_mask]
test_y = test_y[test_mask]

train_x = train_x/255.0
test_x = test_x/255.0

train_y[train_y==0] = 0
train_y[train_y==1] = 1

test_y[test_y==0] = 0
test_y[test_y==1] = 1

train_y = train_y.float()
test_y = test_y.float()

mask_6 = train_y == 0
mask_9 = train_y == 1

train_x_6 = train_x[mask_6]
train_x_9 = train_x[mask_9]

train_y_6 = train_y[mask_6]
train_y_9 = train_y[mask_9]

mask_6 = test_y==0
mask_9 = test_y==1

test_x_6 = test_x[mask_6]
test_x_9 = test_x[mask_9]

test_y_6 = test_y[mask_6]
test_y_9 = test_y[mask_9]

test_y_6[:] = 0.
test_y_9[:] = 1.
train_y_6[:] = 0.
train_y_9[:] = 1.


train_y = torch.cat([train_y_6[0:5000],train_y_9[0:5000]])
train_x = torch.cat([train_x_6[0:5000],train_x_9[0:5000]])

test_x = torch.cat([test_x_6[0:1000],test_x_9[0:1000]])
test_y = torch.cat([test_y_6[0:1000],test_y_9[0:1000]])

train_x = rearrange(train_x.to(device),'b h w c -> b c h w')
train_y = train_y.to(device)

test_x = rearrange(test_x.to(device),'b h w c -> b c h w')
test_y = test_y.to(device)


# In[4]:


def NTK_weights(m): #!!! BTW, resnet wasn't in the NTK parameterization
    if isinstance(m, nn.Linear):
        print(m.weight.shape)
        nn.init.normal_(m.weight.data)#/m.weight.shape[0]
        if m.bias != None:
            nn.init.normal_(m.bias.data)#/m.weight.shape[0]
    if isinstance(m, nn.Conv2d):
        print(m.weight.shape)
        nn.init.normal_(m.weight.data)#/m.weight.shape[0]
        if m.bias != None:
            nn.init.normal_(m.bias.data)#/m.weight.shape[0]


# In[5]:


class Model(torch.nn.Module):
    def __init__(self,WIDTH=8):
        super(Model, self).__init__()
        self.WIDTH = WIDTH
        
        self.conv1 = nn.Conv2d(3, 8*WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(8*WIDTH, 8*WIDTH, 3,padding=1)
        self.pool1 = nn.MaxPool2d(2) #16
        
        self.conv3 = nn.Conv2d(8*WIDTH, 4*WIDTH, 3, padding=1)
        self.conv4 = nn.Conv2d(4*WIDTH, 4*WIDTH, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2) #8
        
        self.conv5 = nn.Conv2d(4*WIDTH, WIDTH, 3, padding=1)
        self.conv6 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2) #4
        
        self.fc1 = nn.Linear(WIDTH*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.conv1(x))/np.sqrt(8*self.WIDTH*3*3)
        y = F.relu(self.conv2(y))/np.sqrt(8*self.WIDTH*3*3)
        y = self.pool1(y)
        
        y = F.relu(self.conv3(y))/np.sqrt(4*self.WIDTH*3*3)
        y = F.relu(self.conv4(y))/np.sqrt(4*self.WIDTH*3*3)
        y = self.pool2(y)
        
        y = F.relu(self.conv5(y))/np.sqrt(self.WIDTH*3*3)
        y = F.relu(self.conv6(y))/np.sqrt(self.WIDTH*3*3)
        y = self.pool3(y)
        
        y = y.reshape(y.shape[0],-1)
        y = F.relu(self.fc1(y))/np.sqrt(128)
        y = F.relu(self.fc2(y))/np.sqrt(64)
        y = self.fc3(y)
        

        return y
        


# In[6]:


def make_model(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    #device='cuda'

    model = Model()

    #model.cuda()
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model.apply(NTK_weights)
    return model


# In[7]:


criterion = torch.nn.BCEWithLogitsLoss()

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_loader = DataLoader(TensorDataset(train_x,train_y),batch_size=64,shuffle=True)
test_loader = DataLoader(TensorDataset(test_x,test_y),batch_size=64,shuffle=False)


# In[8]:


combined_loader = DataLoader(TensorDataset(torch.cat([test_x,train_x]),torch.cat([test_y,train_y])),batch_size=60,shuffle=False)


# In[9]:


def calculate_test_acc(model,test_loader):
    with torch.no_grad():
        accuracies = []
        outputs = []
        for data,label in test_loader:
            data=data.to('cuda')
            label=label.to('cuda')
            output = model(data) - model_init(data)
            acc = torch.sum(label.squeeze()==torch.round(torch.sigmoid(output).squeeze()))
            accuracies.append(acc.item())
            outputs.append(torch.sigmoid(output).squeeze())
    print('Test Acc: {:.4f}'.format(np.sum(accuracies)/2000))
    return torch.cat(outputs)


# In[10]:


N_EPOCHS=200


# In[11]:


for SEED in range(0,100):
    model = make_model(SEED)
    model.to(device)
    
    model_init = make_model(SEED)
    model_init.to(device)
    for names, params in model_init.named_parameters():
        params.requires_grad=False
    
    optimizer = torch.optim.Adam(model.parameters(),1e-3)

    outer_losses = []
    outer_acc = []

    for step in tqdm(range(1,N_EPOCHS+1)):       
        losses=[]
        accuracies=[]
        for data,label in train_loader:
            data = data.to(device)
            target = label.to(device)
            model.zero_grad()

            output = model(data) - model_init(data)
            loss = criterion(output.squeeze(),target.squeeze())
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            acc = torch.sum(torch.sign(target.squeeze()-0.5)==torch.sign(torch.sigmoid(output).squeeze()-0.5))
            accuracies.append(acc.item())
            


        outer_losses.append(np.mean(losses))
        outer_acc.append(np.sum(accuracies)/(10000))

    calculate_test_acc(model,test_loader)    
    plt.plot(outer_losses)
    plt.title('CIFAR2 CNN SEED {} Loss History'.format(SEED))
    #plt.savefig('./images/CNN_CIFAR2_CI_trainhistory/loss_{}'.format(SEED))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(outer_acc)
    plt.title('CIFAR2 CNN SEED {} Train Acc History'.format(SEED))
    #plt.savefig('./images/CNN_CIFAR2_CI_trainhistory/acc_{}'.format(SEED))
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc')
    plt.show()
            
    torch.save(model.state_dict(), '../MODELS/smallmodel/CIFARCNN/CIFAR2_trained_SEED{}.pt'.format(SEED))


# # Load in those models and Calculate the CI NTK

# In[11]:


def calculate_COSSIM(NTK,):
    K0 = NTK[2000::,2000::] #this is the train data gram matrix
    K1 = NTK[0:2000,2000::] #this is the test-data's sim to train data
    K2 = NTK[0:2000,0:2000]

    K1 = K1 / (np.sqrt(np.diagonal(K0))[None,:]) / (np.sqrt(np.diagonal(K2))[:,None])
    K0 = K0 / np.sqrt(np.diagonal(K0))[None,:] / np.sqrt(np.diagonal(K0))[:,None]
        
    return K0, K1

from scipy.stats import binomial

def misclassified_overlap_probability(gamma_hat,N,W,Omega,visualize=False):
    """
    Calculates the probabiliy of the null hypothesis that two uncorreleated models classify the same N points
    incorrectly, with number of incorrect points W from model one and Omega from model 2, by first assuming we 
    are selecting from the W model points to be wrong, and then asking of those which are also wrong in the Omega model.
    If the models are truly uncorreleated, we can model both as seperate Bernoulli trials, governed by the Binomial dist.
    
    Given an observed rate of this overlap gamma_hat, then we can ask how probable given our hypothesis that the models
    are uncorreleated seeing such a value is. I return this value itself, from which one would externally set some alpha
    threshold, or report the p-value directly.
    """
    assert N > W
    assert N > Omega
    
    P_both = (Omega*W)/(N**2)
    P_H0 = 1 - binom.cdf(W*gamma_hat,W,P_both)
    
    if visualize:
        gamma_ = np.linspace(0,1,1000)
        plt.plot(W*gamma_,binom.pmf(W*gamma_,W,P_both,gamma_),label='PMF of B(k,n,p)')
        plt.vlines(W*gamma_hat,0,0.3,linestyle='dashed',color='k',label=r'$W \times \hat{\Gamma}$')
        plt.xlabel(r'Expected number of successes: $W \times \Gamma$',fontsize=14)
        plt.title(r'Visualizing How Improbable $\hat{\Gamma}$ is',fontsize=14)
        plt.ylabel('Probability of Observation',fontsize=14)
        plt.legend()
        plt.show()
    
    return P_H0



# In[13]:


test_y_np = test_y.cpu().numpy()


# In[15]:


from sklearn import svm
###
from sklearn.linear_model import SGDClassifier
###

Ns = []
Ws = []
Omegas = []
gamma_hats = []
Ps = []
test_acc = []
train_acc = []
SEED = []
kSVM_test_acc = []
confidences = []
L2CosSimCorrectClassOnly = []
L2CosSim = []
SEEDs = []
all_correct_kSVM_vectors = []
all_correct_NN_vectors = []
###
LKR_test_acc = []
all_correct_LKR_vectors = []
logits_LKR = []
logits_NN = []
logits_kSVM = []
###

for SEED in tqdm(range(0,100)):
    model = make_model(SEED)
    model.cuda()
    model_init = make_model(SEED)
    model_init.cuda()
    model.load_state_dict(torch.load('../MODELS/smallmodel/CIFARCNN/CIFAR2_trained_SEED{}.pt'.format(SEED)))
    
    NTK = ezntk.vmap_ntk_loader(model,combined_loader)
    NTK_np = 0
    for key in NTK.keys():
        NTK_np+= NTK[key].detach().cpu().numpy().copy()
    del NTK
    NTK = np.load('../KERNELS/smallmodel/CIFARCNN/CIFAR2_NTK{}.npy'.format(SEED))

    
    index_test = len(test_x)
    K0 = NTK[index_test::,index_test::]
    K1 = NTK[0:index_test,index_test::]
    K2 = NTK[0:index_test,0:index_test]
    K1 = K1 / np.sqrt(np.diag(K2))[:,None] / np.sqrt(np.diag(K0))[None,:]
    K0 = K0 / np.sqrt(np.diag(K0))[:,None] / np.sqrt(np.diag(K0))[None,:]
    
    clf = svm.SVC(kernel="precomputed",probability=True)
    clf.fit(K0,train_y.cpu().numpy())
    predictions = clf.predict(K1)
    kSVM_proba = clf.predict_proba(K1)

    with torch.no_grad():
        output = model(test_x) - model_init(test_x)
    predictions_model = np.round(torch.sigmoid(output).detach().cpu().numpy())[:,0]
    
    with torch.no_grad():
        output_train = model(train_x) - model_init(train_x)
    predictions_model_train = np.round(torch.sigmoid(output_train).detach().cpu().numpy())[:,0]
    correct_NN_train = predictions_model_train == train_y.cpu().numpy()

    ###
    LogKernelRegression = SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,fit_intercept=True,class_weight="balanced")
    LogKernelRegression.fit(K0,train_y.cpu().numpy())
    y_LogKernelRegression = LogKernelRegression.predict(K1)
    correct_LKR = y_LogKernelRegression == test_y.cpu().numpy()
    ###
    ###!!!
    weights = LogKernelRegression.coef_
    intercept = LogKernelRegression.intercept_
    logit_LKR = K1@weights.T + intercept
    logits_LKR.append(logit_LKR)
    
    logits_NN.append(torch.sigmoid(output).detach().cpu().numpy())
    ###!!!
    
    
    correct_NN = predictions_model == test_y.cpu().numpy()
    correct_kSVM = predictions == test_y.cpu().numpy()
    
    N = len(correct_NN)
    W = np.sum(~correct_NN)
    Omega = np.sum(~correct_kSVM)
    gamma_hat = gamma_hat = np.logical_and(~correct_NN,~correct_kSVM).sum() / np.logical_or(~correct_NN,~correct_kSVM).sum()
    
    P_h0 = misclassified_overlap_probability(gamma_hat,N,W,Omega,False)
    
    Ns.append(N)
    Ws.append(W)
    Omegas.append(Omega)
    gamma_hats.append(gamma_hat)
    Ps.append(P_h0)
    test_acc.append(np.sum(correct_NN)/N)
    train_acc.append(np.sum(correct_NN_train)/len(correct_NN_train))
    SEEDs.append(SEED)
    kSVM_test_acc.append(np.sum(correct_kSVM)/N)
    confidences.append(torch.sigmoid(output).detach().cpu().numpy())
    L2CosSimCorrectClassOnly.append(np.array([np.linalg.norm(K1[i,(train_y==test_y[i]).cpu().numpy()],2,axis=0) for i in range(len(K1))]))
    L2CosSim.append(np.linalg.norm(K1,2,axis=1))
    all_correct_NN_vectors.append(correct_NN)
    all_correct_kSVM_vectors.append(correct_kSVM)
    ###
    LKR_test_acc.append(np.sum(correct_LKR)/N)
    all_correct_LKR_vectors.append(correct_LKR)
    ###
    logits_kSVM.append(kSVM_proba)


# In[16]:


overlap_misclassified_NN_models = []
for i in range(100):
    for j in range(100):
        if i==j or j<=i:
            continue
        else:
            array1 = all_correct_NN_vectors[i].squeeze()
            array2 = all_correct_NN_vectors[j].squeeze()
            
            overlap_misclassified_NN_model = np.logical_and(~array1,~array2).sum()
            number_incorrect_model1 = (~array1).sum()
            overlap_misclassified_NN_models.append(overlap_misclassified_NN_model/number_incorrect_model1)


# In[17]:


overlap_misclassified_kSVM_models = []
for i in range(100):
    array1 = all_correct_NN_vectors[i]
    array2 = all_correct_kSVM_vectors[i]
    
    overlap_misclassified_kSVM_model = np.logical_and(~array1,~array2).sum()
    number_incorrect_model1 = (~array1).sum()
    overlap_misclassified_kSVM_models.append(overlap_misclassified_kSVM_model/number_incorrect_model1)


# In[18]:


report = {}

report['N'] = Ns
report['W'] = Ws
report['Omega'] = Omegas
report['gamma'] = gamma_hats
report['P'] = Ps
report['test_acc_NN'] = test_acc
report['train_acc_NN'] = train_acc
report['test_acc_kSVM'] = kSVM_test_acc
report['confidences_NN'] = confidences
report['L2CosSim_corr'] = L2CosSimCorrectClassOnly
report['L2CosSim'] = L2CosSim
report['SEED'] = SEEDs
report['overlap_misclassified_NN_models'] = overlap_misclassified_NN_models
report['overlap_mislassified_kSVM_models'] = overlap_misclassified_kSVM_models
report['correct_NN_vector'] = all_correct_NN_vectors
report['correct_kSVM_vector'] = all_correct_kSVM_vectors
###
report['test_acc_LKR'] = LKR_test_acc
report['correct_LKR_vector'] = all_correct_LKR_vectors
report['logits_LKR'] = logits_LKR
report['logits_NN'] = logits_NN
report['logits_kSVM'] = logits_kSVM
###
print('would save!')
#np.save('../experimentresults/CIFAR2_CNN_100_models_results.npy',report)
