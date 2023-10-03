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

#from numba import njit

from tqdm import tqdm

import os

import gc

import torchntk.autograd as ezntk
import torchvision.models as models
from tqdm import tqdm


# In[2]:


SEED = 0
N_EPOCHS=200
LR=5e-3
device = 'cuda'


# In[3]:


train_data = datasets.FashionMNIST(
    root = '../../DATA/',
    train = True,                          
    download = True,            
)


test_data = datasets.FashionMNIST(
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

train_x = train_x.to(device).reshape(-1,1,28,28)
train_y = train_y.to(device)

test_x = test_x.to(device).reshape(-1,1,28,28)
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
        
        self.conv1 = nn.Conv2d(1, 8*WIDTH, 3, padding=1)
        #self.conv2 = nn.Conv2d(8*WIDTH, 8*WIDTH, 3,padding=1)
        self.pool1 = nn.AvgPool2d(2) #14
        
        self.conv3 = nn.Conv2d(8*WIDTH, 4*WIDTH, 3, padding=1)
        #self.conv4 = nn.Conv2d(4*WIDTH, 4*WIDTH, 3, padding=1)
        self.pool2 = nn.AvgPool2d(2) #7
        
        self.conv5 = nn.Conv2d(4*WIDTH, WIDTH, 3, padding=1)
        #self.conv6 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.pool3 = nn.AvgPool2d(2) #3, 4?
        
        self.fc1 = nn.Linear(WIDTH*3*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.conv1(x))/np.sqrt(8*self.WIDTH*3*3)
        y = F.relu(self.pool1(y))
        
        y = F.relu(self.conv3(y))/np.sqrt(4*self.WIDTH*3*3)
        y = F.relu(self.pool2(y))
        
        y = F.relu(self.conv5(y))/np.sqrt(self.WIDTH*3*3)
        y = F.relu(self.pool3(y))
        
        y = y.view(y.shape[0],-1)
        y = F.relu(self.fc1(y))/np.sqrt(128)
        y = F.relu(self.fc2(y))/np.sqrt(64)
        y = self.fc3(y)
        
        return y
        
    def set_feature_maps(self,x):
        with torch.no_grad():
        
            y1 = F.relu(self.conv1(x))/np.sqrt(8*self.WIDTH*3*3)
            y2 = F.relu(self.pool1(y1))

            y3 = F.relu(self.conv3(y2))/np.sqrt(4*self.WIDTH*3*3)
            y4 = F.relu(self.pool2(y3))

            y5 = F.relu(self.conv5(y4))/np.sqrt(self.WIDTH*3*3)
            y6 = F.relu(self.pool3(y5))

            y7 = y6.view(y6.shape[0],-1)
            y8 = F.relu(self.fc1(y7))/np.sqrt(128)
            y9 = F.relu(self.fc2(y8))/np.sqrt(64)
            
            self.y1 = y1
            self.y2 = y2
            self.y3 = y3
            self.y4 = y4
            self.y5 = y5
            self.y7 = y7
            self.y8 = y8
            self.y9 = y9
            #only parameterized layers go into maps, and skip last layer of course
            self.maps = [y1,y3,y5,y8,y9]
            


def make_model(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    #device='cuda'

    model = Model()

    model.to(device)
    model.apply(NTK_weights)
    return model


# In[9]:


model = make_model(0)


# In[10]:


criterion = torch.nn.MSELoss()

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

#NTK_w_vmap = vmap_ntk_loader(model,xloader)
#tgts = torch.zeros(HOW_MANY,device='cuda')
train_loader = DataLoader(TensorDataset(train_x,train_y),batch_size=4)
test_loader = DataLoader(TensorDataset(test_x,test_y),batch_size=4)


# In[11]:


combined_loader = DataLoader(TensorDataset(torch.cat([test_x,train_x]),torch.cat([test_y,train_y])),batch_size=60,shuffle=False)

#model = make_model(0)
#model.load_state_dict(torch.load('./MODELS/FMNIST_manymodels/FMNIST_trained_SEED{}.pt'.format(SEED)))


# # Load in those models and Calculate the CI NTK

# In[12]:


def calculate_COSSIM(NTK,):
    K0 = NTK[2000::,2000::] #this is the train data gram matrix
    K1 = NTK[0:2000,2000::] #this is the test-data's sim to train data
    K2 = NTK[0:2000,0:2000]

    K1 = K1 / (np.sqrt(np.diagonal(K0))[None,:]) / (np.sqrt(np.diagonal(K2))[:,None])
    K0 = K0 / np.sqrt(np.diagonal(K0))[None,:] / np.sqrt(np.diagonal(K0))[:,None]
        
    return K0, K1


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
    plt.title('FMNIST2 CNN SEED {} Loss History'.format(SEED))
    #plt.savefig('./images/CNN_CIFAR2_CI_trainhistory/loss_{}'.format(SEED))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(outer_acc)
    plt.title('FMNIST2 CNN SEED {} Train Acc History'.format(SEED))
    #plt.savefig('./images/CNN_CIFAR2_CI_trainhistory/acc_{}'.format(SEED))
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc')
    plt.show()
            
    torch.save(model.state_dict(), '../MODELS/smallmodel/FMNISTCNN/FMNIST2_trained_SEED{}.pt'.format(SEED))
    




def autograd_J_layer(model: torch.nn.Module, y :torch.Tensor, param_name :str, device :str='cuda'):
    """calulcates each layerwise component and returns a torch.Tensor representing the NTK

        parameters:
            model: a torch.nn.Module object. Must terminate to a single neuron output
            y: the final single neuron output of the model evaluated on some data

        returns:
            NTK: a torch.Tensor representing the emprirical neural tangent kernel of the model
    """
    #Formatting, make sure y is the correct shape. Currently, we only support single outputs.
    if len(y.shape) > 2:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(y.shape))
    if len(y.shape) == 2:
        if y.shape[1] != 1:
            raise ValueError('y must be 1-D, but its shape is: {}'.format(y.shape))
        else:
            y = y[:,0] #smash down to 1D.


    #Now calculate the Jacobian using torch.autograd:
    #how do we parallelize this operation across multiple gpus or something? that be sweet.
    for i,z in enumerate(model.named_parameters()):
        name, param = z
        if name!=param_name:
            continue
        this_grad=[]
        for i in range(len(y)): #first dimension must be the batch dimension
            model.zero_grad()
            y[i].backward(retain_graph=True,inputs=param) #multiple backward calls require retain_graph=True
            this_grad.append(param.grad.detach().reshape(-1).clone()) #cloning is neccessary or else the underlying
            param.grad = None #apparently this helps.
            #data we point to gets updated.

        J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
        del this_grad

    return J_layer

from scipy.stats import binom
def misclassified_overlap_probability(gamma_hat: float,N: int,W: int,Omega: int, visualize: bool=False):
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


# In[14]:


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
all_correct_NN_vectors = []
all_correct_kSVM_vectors = []
###
LKR_test_acc = []
all_correct_LKR_vectors = []
logits_LKR = []
logits_NN = []
logits_kSVM = []
###

for SEED in tqdm(range(0,100)):
    model.load_state_dict(torch.load('../MODELS/smallmodel/FMNISTCNN/FMNIST2_trained_SEED{}.pt'.format(SEED)))
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device='cuda'

    evalK, index_test = compute_eval_kernel(model,test_x,test_y,train_x,train_y)
    NTK = torch.sum(torch.stack([evalK[key] for key in evalK.keys()]),dim=0)
    del evalK
    np.save('../KERNELS/smallmodel/FMNISTCNN/FMNIST2_NTK{}.npy'.format(SEED),NTK.cpu().numpy())
    del NTK
    NTK = np.load('../KERNELS/smallmodel/FMNISTCNN/FMNIST2_NTK{}.npy'.format(SEED))
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
        output = model(test_x)
    predictions_model = np.round(torch.sigmoid(output).detach().cpu().numpy())[:,0]
    
    with torch.no_grad():
        output_train = model(train_x)
    predictions_model_train = np.round(torch.sigmoid(output_train).detach().cpu().numpy())[:,0]
    correct_NN_train = predictions_model_train == train_y.cpu().numpy()

    correct_NN = predictions_model == test_y.cpu().numpy()
    correct_kSVM = predictions == test_y.cpu().numpy()
    
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
    
    N = len(correct_NN)
    W = np.sum(~correct_NN)
    Omega = np.sum(~correct_kSVM)
    gamma_hat = np.logical_and(~correct_NN,~correct_kSVM).sum() / np.logical_or(~correct_NN,~correct_kSVM).sum()
    
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

    #print('Gamma = {}, N = {}, W = {}, Omega = {}'.format(gamma_hat,N,W,Omega))
    #print('Probability of the Null Hypothesis: {:.1e}'.format(misclassified_overlap_probability(gamma_hat,N,W,Omega,True)))
    


# In[15]:


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


# In[16]:


overlap_misclassified_kSVM_models = []
for i in range(100):
    array1 = all_correct_NN_vectors[i]
    array2 = all_correct_kSVM_vectors[i]
    
    overlap_misclassified_kSVM_model = np.logical_and(~array1,~array2).sum()
    number_incorrect_model1 = (~array1).sum()
    overlap_misclassified_kSVM_models.append(overlap_misclassified_kSVM_model/number_incorrect_model1)


# In[17]:


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
report['L2CosSim'] = L2CosSim
report['L2CosSim_corr'] = L2CosSimCorrectClassOnly
report['SEED'] = SEEDs
report['correct_NN_vector'] = all_correct_NN_vectors
report['correct_kSVM_vector'] = all_correct_kSVM_vectors
###
report['test_acc_LKR'] = LKR_test_acc
report['correct_LKR_vector'] = all_correct_LKR_vectors
report['logits_LKR'] = logits_LKR
report['logits_NN'] = logits_NN
###
report['logits_kSVM'] = logits_kSVM

np.save('../experimentresults/FMNIST2_CNN_100_models_results.npy',report)


