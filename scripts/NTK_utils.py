
#Notes: an adaptive batch size is infeasible because the largest memory component is the model. Even if you could in principle calculate the backware propogation at once with 

import torch
from torch import nn, optim
from torch import load
from torch.nn import functional as F
from torch import autograd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import glob

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

def construct_layerntk_from_components(name :str,
                                       batch_size :int,
                                       cache_directory :str='./testDDP',
                                       device='cuda'):
    """
    name examples:
    'bert.encoder.layer.1.attention.output.LayerNorm.bias-jlayer-*.pt'
    
    This reconstructs the NTK from a directory of batched Jacobian calculations
    
    ASSUMES THAT ALL BATCHES HAVE THE SAME BATCH SIZE, should fail hard.
    """
    
    filepaths = glob.glob(os.path.join(cache_directory,name))
    #assert len(filepaths)==world_size * N
    how_many_files = len(filepaths)
    NTK = torch.zeros(batch_size*how_many_files,batch_size*how_many_files,device=device)
    #lower_tril_mask = torch.ones(batch_size*how_many_files,batch_size*how_many_files,dtype=bool).tril()
    
    for i,path in enumerate(filepaths):
        #assumes that -jlayer- is this reserved string, cant be the name of a layer
        path_ = path.split('-jlayer-')[1]
        path_ = path_.split('.pt')[0]
        rank0,step0 = path_.split('-')#this is neat,  I only thought it worked on tuples.
        rank0=int(rank0)#cast str to int
        step0=int(step0)

        J_0 = torch.load(path) #load on the main gpu

        for j,path in enumerate(filepaths):
            path_ = path.split('-jlayer-')[1]
            path_ = path_.split('.pt')[0]
            rank1,step1 = path_.split('-')#this is neat,  I only thought it worked on tuples.
            rank1=int(rank1)#cast str to int
            step1=int(step1)
            if j<i:
                continue
            J_1 = torch.load(path)
            NTK_component = torch.mm(J_0.to(J_1.get_device()), J_1.T)

            #Based upon how the DistributedSampler chooses indices:
            index00 = step0*1*batch_size + rank0
            index01 = (step0+1)*1*batch_size

            index10 = step1*1*batch_size + rank1
            index11 = (step1+1)*1*batch_size
            
            NTK[index00:index01:1,index10:index11:1] = NTK_component
            NTK[index10:index11:1,index00:index01:1] = NTK_component.T
    
    return NTK.cpu()

def sum_up_NTK_additive_components(additivecomponents_directory :str='./testDDP_additivecomponents/*.pt'):
    filepaths = glob.glob(additivecomponents_directory)
    for i,file in enumerate(filepaths):
        if i == 0:
            NTK = torch.load(file)
        else:
            NTK+= torch.load(file)

    return NTK

    
    
def construct_components(model, layername, outneuron, batch_size, directory, data_inputs, data_labels, data_masks):
    """
    model:torch.nn.Module object already on a GPU
    
    name: the name of the layer as a str
    
    data_inputs, labels, masks: input transformer data
    
    batch_size: batch size for the operation, for BERT 128 is as high as we can go on an A100 GPU
        in principle, you should make batchsize as large as possible since we aren't training
    """
    #First, lets set up a DataLoader object
    small_data = TensorDataset(data_inputs, data_labels, data_masks)
    dataloader = DataLoader(small_data, batch_size=batch_size, shuffle=False,drop_last=True)
    
    for step,batch in enumerate(dataloader):#these are the chunks. need to know the chunks.

            #can only reliably handle training loops generated like this...
            #so in principle, you will always need to modify this file. :(
            b_input_ids = batch[0].to('cuda')
            b_input_mask = batch[2].to('cuda')
            b_labels = batch[1].to('cuda')
            
            #Zero the gradient
            for param in model.parameters():
                param.grad = None

            outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
            output = outputs[1]
            y = output[:,outneuron] #1D

            J_layer = autograd_J_layer(model, y, layername)
            torch.save(J_layer,os.path.join(directory,'{}-{}-jlayer-{}-{}.pt'.format(layername,outneuron,0,step)))
            del J_layer
            del y
            del output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
