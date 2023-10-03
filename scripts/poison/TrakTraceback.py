#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from pathlib import Path
from torchvision import datasets

from einops import rearrange

import torch
import pickle

from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from einops import rearrange
import os
import numpy as np


# In[2]:


#We are using 1 model because we have only 1 model?
ckpts = [torch.load('../MODELS/poison/poisoned_CNN.pt'),]


# In[3]:


from trak import TRAKer


# In[4]:


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


# In[5]:


model = Model()
model.to('cuda').eval()
#torch.manual_seed(1234)

optim = torch.optim.SGD(model.parameters(),1e-2,momentum=0.9,nesterov=True)
loss = torch.nn.CrossEntropyLoss()


# In[6]:


traker = TRAKer(model=model,
                task='image_classification',
                train_set_size=50_000,
                proj_dim=2048)


# In[7]:


with open('../DATA/poison/cifar_cifar2_res.p','rb') as f:
    RES = pickle.load(f)


# In[8]:


RES.keys()

train_x = rearrange(torch.tensor(RES['injected_X'],dtype=torch.float32),'b h w c -> b c h w')
train_y = torch.argmax(torch.tensor(RES['injected_Y'],dtype=torch.long),axis=1)

test_x = rearrange(torch.tensor(RES['injected_X_test'],dtype=torch.float32),'b h w c -> b c h w')
test_y = torch.tensor(RES['injected_Y_test'],dtype=torch.float32)

test_x_og = rearrange(torch.tensor(RES['X_test'],dtype=torch.float32),'b h w c -> b c h w')
test_y_og = torch.tensor(RES['Y_test'],dtype=torch.float32)


test_x_all = rearrange(torch.cat([torch.tensor(RES['injected_X_test'],dtype=torch.float32),torch.tensor(RES['X_test'],dtype=torch.float32)]),'b h w c -> b c h w')
test_y_all = torch.cat([torch.tensor(RES['injected_Y_test'],dtype=torch.long),torch.tensor(RES['Y_test'],dtype=torch.long)])

test_y_all = torch.tensor(torch.argmax(test_y_all,axis=1),dtype=torch.long)

train_loader = DataLoader(TensorDataset(train_x,train_y),batch_size=100,shuffle=False)
test_loader = DataLoader(TensorDataset(test_x,test_y),batch_size=1,shuffle=False)
test_loader_og = DataLoader(TensorDataset(test_x_og,test_y_og),batch_size=1,shuffle=False)


test_loader_all = DataLoader(TensorDataset(test_x_all,test_y_all),batch_size=100,shuffle=False)


# In[9]:


from tqdm import tqdm

for model_id, ckpt in enumerate(ckpts):
    # TRAKer loads the provided checkpoint and also associates
    # the provided (unique) model_id with the checkpoint.
    traker.load_checkpoint(ckpt, model_id=1)

    for batch in tqdm(train_loader):
        batch = [x.cuda() for x in batch]
        
        # TRAKer computes features corresponding to the batch of examples,
        # using the checkpoint loaded above.
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

# Tells TRAKer that we've given it all the information, at which point
# TRAKer does some post-processing to get ready for the next step
# (scoring target examples).
traker.finalize_features()


# In[10]:


for model_id, checkpoint in enumerate(ckpts):
    traker.start_scoring_checkpoint(checkpoint, model_id=1, num_targets=len(test))
    for batch in test_loader_all:
        #batch[1] = torch.tensor(batch[1],dtype=torch.long)
        traker.score(batch=batch,num_samples=batch[0].shape[0])


# In[ ]:


scores = traker.finalize_scores()


# In[ ]:


for model_id, checkpoint in enumerate(ckpts):
    traker.start_scoring_checkpoint(checkpoint, model_id=1, num_targets=len(train))
    for batch in train_loader:
        batch[1] = torch.tensor(batch[1],dtype=torch.long)
        traker.score(batch=batch,num_samples=batch[0].shape[0])
        
scores_train = traker.finalize_scores()


# In[ ]:




