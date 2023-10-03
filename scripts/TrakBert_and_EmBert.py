#!/usr/bin/env python
# coding: utf-8

# In[4]:


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

from importlib import reload
from tqdm import tqdm


# In[5]:


#We are using 1 model because we have only 1 model?
ckpts = [torch.load('/rcfs/projects/task0_pmml/BERT/model_frozen.pt'),
        torch.load('/rcfs/projects/task0_pmml/BERT/one_gpu_development/MANY_BERT_MODELS/BERT-base_SEED1.pt'),
        torch.load('/rcfs/projects/task0_pmml/BERT/one_gpu_development/MANY_BERT_MODELS/BERT-base_SEED2.pt'),
        torch.load('/rcfs/projects/task0_pmml/BERT/one_gpu_development/MANY_BERT_MODELS/BERT-base_SEED3.pt')]


# In[ ]:





# In[6]:


from transformers import BertForSequenceClassification, AdamW, BertConfig


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,   
    output_attentions = False,
    output_hidden_states = False,
)

model.to('cuda').eval()


# In[7]:


import pandas as pd
# because the dataset is int tsv format we have to use delimeter.
df = pd.read_csv("../cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_sources', 'label', 'label_note', 'sentence'])

# creating a copy so we don't messed up our original dataset.
data=df.copy()

data.drop(['sentence_sources','label_note'],axis=1,inplace=True)
sentences=data.sentence.values
labels = data.label.values

from transformers import BertTokenizer
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
    
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 128

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN , truncating="post", padding="post")

attention_masks = []

for sent in input_ids:
    
    # Generating attention mask for sentences.
    #   - when there is 0 present as token id we are going to set mask as 0.
    #   - we are going to set mask 1 for all non-zero positive input id.
    att_mask = [int(token_id > 0) for token_id in sent]
    
   
    attention_masks.append(att_mask)
    
from sklearn.model_selection import train_test_split

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=0)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,test_size=0.2, random_state=0)

#changing the numpy arrays into tensors for working on GPU. 
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Deciding the batch size for training.

batch_size = 32

#DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, shuffle=False)

N_test_DATAPOINTS = 8192 - len(train_inputs)

# DataLoader for our validation(test) set.
validation_data = TensorDataset(validation_inputs[0:N_test_DATAPOINTS], validation_masks[0:N_test_DATAPOINTS], validation_labels[0:N_test_DATAPOINTS])
validation_labels = validation_labels[0:N_test_DATAPOINTS]
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, shuffle=False)


# # Start Trak in Earnest

# In[ ]:


import trak
TRAKer = trak.TRAKer


# In[ ]:


traker = TRAKer(model=model,
                task='text_classification',
                train_set_size=len(train_inputs),
                proj_dim=1024)


# In[7]:




for model_id, ckpt in enumerate(ckpts):
    # TRAKer loads the provided checkpoint and also associates
    # the provided (unique) model_id with the checkpoint.
    traker.load_checkpoint(ckpt, model_id=model_id)

    for batch in tqdm(train_dataloader):
        batch = [x.cuda() for x in batch]
        input_shape = batch[0].shape
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device='cuda')
        batch = [batch[0], token_type_ids, batch[1], batch[2]]
        # TRAKer computes features corresponding to the batch of examples,
        # using the checkpoint loaded above.
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

# Tells TRAKer that we've given it all the information, at which point
# TRAKer does some post-processing to get ready for the next step
# (scoring target examples).
traker.finalize_features()


# In[9]:


for model_id, checkpoint in enumerate(ckpts):
    traker.start_scoring_checkpoint(checkpoint, model_id=model_id, num_targets=len(validation_data))
    for batch in validation_dataloader:
        input_shape = batch[0].shape
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device='cuda')
        batch = [batch[0], token_type_ids, batch[1], batch[2]]
        batch = [x.cuda() for x in batch]
        #batch[1] = torch.tensor(batch[1],dtype=torch.long)
        traker.score(batch=batch,num_samples=batch[0].shape[0])


# In[10]:


scores = traker.finalize_scores()


# In[12]:


for model_id, checkpoint in enumerate(ckpts):
    traker.start_scoring_checkpoint(checkpoint, model_id=model_id, num_targets=len(train_data))
    for batch in train_dataloader:
        input_shape = batch[0].shape
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device='cuda')
        batch = [batch[0], token_type_ids, batch[1], batch[2]]
        batch = [x.cuda() for x in batch]
        #batch[1] = torch.tensor(batch[1],dtype=torch.long)
        traker.score(batch=batch,num_samples=batch[0].shape[0])
        
scores_train = traker.finalize_scores()


# In[ ]:



# # Embedding Bert-- only run if you didnt run Trak, Trak eats too much memory

# In[31]:


batch_size=32

all_inputs = torch.cat([train_inputs,validation_inputs[0:N_test_DATAPOINTS]]).cuda()
all_masks = torch.cat([train_masks,validation_masks[0:N_test_DATAPOINTS]]).cuda()
all_labels = torch.cat([train_labels,validation_labels[0:N_test_DATAPOINTS]]).cuda()

all_data = TensorDataset(all_inputs, all_masks, all_labels)
all_data_sampler = SequentialSampler(all_data)
all_dataloader = DataLoader(all_data, sampler=all_data_sampler, batch_size=batch_size, shuffle=False)


# In[51]:


ALL_NAMES = []
for name,module in model.named_modules():
    print(name)
    if 'activation' in name:
        continue
    if 'dropout' in name:
        continue
    if 'bert'==name:
        continue
    if 'bert.encoder'==name:
        continue
    if 'bert.encoder.layer'==name:
        continue
    if 'relu' in name:
        continue
    if '' == name:
        continue
    #if len(name.split('.')) < 5 and 'bert.encoder.layer' in name:
    #    continue
        
    ALL_NAMES.append(name)
    


# In[23]:


model.hooks = {}


# In[66]:


for key in list(model.hooks.keys()):
    model.hooks[key].remove()


# In[67]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        if type(output) is tuple:
            if len(output)==1:
                output = output[0]
            else:
                print(output[0].shape)
                print(output[1].shape)
                output = output[0]
        activation[name] = output.detach()
    return hook


# In[70]:


all_inputs


# In[72]:


for modelnum in range(4):
    outer_Em_Kernel = 0
    print('starting: ',modelnum)
    model.load_state_dict(ckpts[modelnum])
    model.eval()
    for k,NAME in tqdm(enumerate(ALL_NAMES)):
        if os.path.exists(f'/rcfs/projects/task0_pmml/BERT/Em_kernel_components/{modelnum}/{NAME}-{k}.pt'):
            continue
        EM_Component = torch.zeros((len(all_masks),len(all_masks)),device='cpu')
        
        for name, module in model.named_modules():
            if name == NAME:
                model.hooks[NAME] = module.register_forward_hook(get_activation(NAME))
        
        with torch.no_grad():
            for i in range(8):
                activation = {}
                outputs = model(all_inputs[i*1024:(i+1)*1024], 
                                token_type_ids=None, 
                                attention_mask=all_masks[i*1024:(i+1)*1024], 
                                labels=all_labels[i*1024:(i+1)*1024],
                                output_hidden_states=False)
                X1_activation = activation[NAME].reshape(1024,-1)
                for j in range(8):
                    activation = {}
                    outputs = model(all_inputs[j*1024:(j+1)*1024], 
                                    token_type_ids=None, 
                                    attention_mask=all_masks[j*1024:(j+1)*1024], 
                                    labels=all_labels[j*1024:(j+1)*1024],
                                    output_hidden_states=False)
                    X2_activation = activation[NAME].reshape(1024,-1)



                    component = torch.matmul(X1_activation,X2_activation.T).cpu()
                    EM_Component[i*1024:(i+1)*1024,j*1024:(j+1)*1024] = component
            outer_Em_Kernel+= EM_Component
            torch.save(EM_Component,f'/rcfs/projects/task0_pmml/BERT/Em_kernel_components/{modelnum}/{NAME}-{k}.pt')
            model.hooks[NAME].remove()
    #torch.save(outer_Em_Kernel,f'/rcfs/projects/task0_pmml/BERT/Em_kernels/seed{modelnum}.pt')


# In[10]:


# for i in range(4):
#     if i==0:
#         continue
#     print('starting: ',i)
#     model.load_state_dict(ckpts[i])

#     Em = 0
#     for layer_num in tqdm(range(13)): #number of attention heads + 1 = 13
#         all_embeddings = []
#         with torch.no_grad():
#             for data in all_dataloader:
#                 inputs, masks, labels = data
#                 inputs = inputs.cuda()
#                 masks = masks.cuda()
#                 labels = labels.cuda()

#                 outputs = model(inputs, 
#                             token_type_ids=None, 
#                             attention_mask=masks, 
#                             labels=labels,
#                             output_hidden_states=True)

#                 all_embeddings.append(outputs[2][layer_num].view(batch_size,-1).cpu())
#                 del outputs
                
#         all_embeddings = torch.cat(all_embeddings).cuda()
#         Em += torch.matmul(all_embeddings,all_embeddings.T).detach().cpu().numpy()
    
#     if i>=1:
#         np.save(f'/rcfs/projects/task0_pmml/BERT/Em_kernels/{i-1}.npy',Em)
#     else:
#         np.save(f'/rcfs/projects/task0_pmml/BERT/Em_kernels/seedless.npy',Em)

