#This script is intended to be run once for each layer on each GPU. 
#That way, each layer can go at its own pace individually of all the other layers.

##NOTE this ends up being terrible inefficient compared to calculating the NTK for all params

import sys

import argparse
import numpy as np
import torch
import random

import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torch import load
from torch.nn import functional as F
from torch import autograd
#import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision import datasets

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import time

import sys
from pathlib import Path

from numba import njit
import pandas as pd

import os
import shutil

import gc

import glob

from tqdm import tqdm

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig

#from torch.nn.parallel import DistributedDataParallel as DDP
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from NTK_utils import construct_layerntk_from_components, construct_components

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    device_ids = list(range(torch.cuda.device_count()))
    gpus = len(device_ids)
    print('Num GPUS: ',gpus)
    print('device ids: ',device_ids)
    #print('GPU detected')
else:
    DEVICE = torch.device("cpu")
    print('No GPU. switching to CPU')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate One Layer of the NTK of BERT on COLA')
    parser.add_argument('layername', metavar='N', type=str, help='the stringname of the layer')
    parser.add_argument('savedir', metavar='M', type=str,help='the string of the cachedir')

    args = parser.parse_args()
    
    #OK, First need to process the dataset.
    ##########################################
    # because the dataset is int tsv format we have to use delimeter.
    df = pd.read_csv("./../cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_sources','label','label_note','sentence'])
    data=df.copy()
    data.drop(['sentence_sources','label_note'],axis=1,inplace=True)
    sentences=data.sentence.values
    labels = data.label.values
    

    # using the low level BERT for our task.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    input_ids = []
    for sent in sentences:
        # so basically encode tokenizing , mapping sentences to thier token ids after adding special tokens.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence which are encoding.
                            add_special_tokens = True, # Adding special tokens '[CLS]' and '[SEP]'
                             )
        input_ids.append(encoded_sent)
    
    MAX_LEN = 128
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN , truncating="post", padding="post")
    attention_masks = []
    for sent in input_ids:
        # Generating attention mask for sentences.
        #   - when there is 0 present as token id we are going to set mask as 0.
        #   - we are going to set mask 1 for all non-zero positive input id.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=0)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=0)
    
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    print('Training Dataset Length: ',len(train_inputs))
    print('Test Dataset Length: ',len(validation_inputs))
    ##########################################
    torch.manual_seed(1)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False)    
    
    model.to('cuda')

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8)    

    train_data = TensorDataset(train_inputs, train_labels, train_masks)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)    

    total_loss = 0
    # putting model in traing mode there are two model eval and train for model
    #model.train()
    #device='cuda'
    #for epoch in range(10):
    #    for step, batch in enumerate(train_dataloader):
    #        optimizer.zero_grad()
    #        #getting ids,mask,labes for every batch
    #        b_input_ids = batch[0].to(device)
    #        b_input_mask = batch[2].to(device)
    #        b_labels = batch[1].to(device)
    #    
    #        outputs = model(b_input_ids, 
    #                token_type_ids=None, 
    #                attention_mask=b_input_mask, 
    #                labels=b_labels)
    #
    #        loss = outputs[0]
    #
    #        # doing back propagation
    #        loss.backward()
    #
    #        optimizer.step()

    #First, fill up testDDP with all the pieces of Jacobian from one layer
    data_inputs = torch.cat([train_inputs,validation_inputs])
    data_labels = torch.cat([train_labels,validation_labels])
    data_masks = torch.cat([train_masks,validation_masks])

    #torch.save(model.state_dict(),'./../model_frozen2.pt')
    #print('finished_model_training2')
    #raise Exception
    #model.to('cpu') #move the model back to CPU or else suffer OOM 
    model.load_state_dict(torch.load('./../model_frozen2.pt'))
    model.to('cuda')
    start_time = time.time()
    for name,param in model.named_parameters():
        if name != args.layername:
            continue
        for i in range(2): #number of outputs, in BERT-BASE =2
            if os.path.exists('./testDDP_additivecomponents/{}-{}.pt'.format(name,i)):
                continue        
            
            #create the directory we will cache the Jacobian components to:
            if not(os.path.exists(args.savedir)):
                os.mkdir(args.savedir)
            
            #calculate the batched Jacobian, save to directory using below style.
            start_construct = time.time()
            construct_components(model, name, i, 128, args.savedir, data_inputs, data_labels, data_masks)
            print('time for constuct components: {} sec'.format(time.time() - start_construct))
            
            #Next, compute the layerwise NTK from that layer. function returns object on CPU
            start_NTK_calc = time.time()
            NTK_component = construct_layerntk_from_components('{}-{}-jlayer-*.pt'.format(name,i),
                                                               128,
                                                               args.savedir,
                                                               'cuda')
            print('time for NTK calc: {} sec'.format(time.time() - start_NTK_calc))

            #save it
            torch.save(NTK_component,'./testDDP_additivecomponents/{}-{}.pt'.format(name,i))

            #free the memory for the next loop, potentially NTK_component is many GB.
            del NTK_component

            #delete the testDDP directories' contents.
            folder = args.savedir
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            os.rmdir(args.savedir)
    print('complete')

    













