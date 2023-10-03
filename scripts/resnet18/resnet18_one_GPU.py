#This script is intended to be run once for each layer on each GPU. That way, each layer can go at it own pace individually of all the other layers.

BATCH_SIZE=300

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

from sklearn.model_selection import train_test_split
from utils2 import process_Cifar10
from utils2 import resnet18
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
    PATH = './../MODELS/resnet18.pt' #!!!
    parser = argparse.ArgumentParser(description='Calculate One Layer of the NTK of BERT on COLA')
    parser.add_argument('layername', metavar='N', type=str, help='the stringname of the layer')
    parser.add_argument('savedir', metavar='M', type=str,help='the string of the cachedir')

    args = parser.parse_args()
    
    #OK, First need to process the dataset.
    ##########################################
    
    train_dataset, test_dataset, combined_dataset = process_Cifar10('./../DATA/')

    print('Training Dataset Length: ',len(train_dataset))
    print('Test Dataset Length: ',len(test_dataset))
    ##########################################
    #setup model and place on GPU
    model = resnet18()#!!!
    model.load_state_dict(torch.load(PATH))
    model.to('cuda')
    model.eval()
    ##########################################
    start_time = time.time()

    data_inputs = combined_dataset.tensors[0]
    data_labels = combined_dataset.tensors[1]

    for name,param in model.named_parameters():
        if name != args.layername:
            continue
        for i in range(10): #number of outputs, in BERT-BASE =2
            if os.path.exists('./testDDP_additivecomponents/{}-{}.pt'.format(name,i)):
                continue        
            
            #create the directory we will cache the Jacobian components to:
            if not(os.path.exists(args.savedir)):
                os.mkdir(args.savedir)

            #calculate the batched Jacobian, save to directory using below style.
            start_construct = time.time()
            construct_components(model, name, i, BATCH_SIZE, args.savedir, data_inputs, data_labels)
            print('time for constuct components: {} sec'.format(time.time() - start_construct))
            
            #Next, compute the layerwise NTK from that layer. function returns object on CPU
            start_NTK_calc = time.time()
            NTK_component = construct_layerntk_from_components('{}-{}-jlayer-*.pt'.format(name,i),
                                                               BATCH_SIZE,
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

    













