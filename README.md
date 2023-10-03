[![arXiv](https://img.shields.io/badge/arXiv-2303.14186-b31b1b.svg?style=flat-square)](XXX)


# Projection NTK: a fork of TRAK

In our [paper](XXX), we introduce projection varients of approximate neural tangent kernel (NTK).
These NTK are computed from Jacobians of neural network models. They benefit from the insight made
in Park 2023 (TRAK), long vectors will retain most of their relative information when projected down
to a smaller feature dimension. We can utilize this to reduce the scaling with number of parameters
in NTK computation, and infact, can tune the computational scaling by choosing the projection dimension.
We observed that for a 1000x reduction via random projection in number of model parameters on ResNet18,
we could calculate an approximate NTK called the projection trace-NTK, that was promising as a surrogate
model for the original neural network and whose residuals with respect to the full trace-NTK fell away
exponentially, see figure below.

![Main figure](/docs/assets/residualdecay.png)

The point of this is that projections enable calculating approximate NTK for large models and large datasets
faster than ever before; with a few tweaks to the underlying TRAK module we can enable PyTorch users to 
evaluate how close their own neural network models are to kernel machines. In addition, the speed and memory
savings should enable exciting new applciations for NTK research. One we demonstrate in the paper is finding
the top 5-most similar images for any test image, see below and in paper for more examples.

![Second figure](/docs/assets/5mostsimilar.png)

We can not overstate how much this work was enabled by TRAK. The goal for this repository is to freeze
a copy to make our work reproducible, but ultimately, we would like to merge our changes back into TRAK. 

## Usage

We provide a rough sketch of usage to calculate the projection-trNTK of a neural network model on Cifar10.

### Make a `TRAKer` instance

```python
from trak import TRAKer

#PyTorch nn.Module object
model = ...

#this is a common state dictionary file for model
checkpoint = torch.load('./checkpoint.pt') 

#we want a dataloader object that combines BOTH train and test data, with shuffle=False
train_and_test_loader = ...
Ndata = len(train_and_test_loader)

#set the projection dimension. we used K=10240 for a ResNet18 with ABC number of model
#parameters. There is assumedly a computation/accuracy tradeoff for K, the probably
#is in some ratio to number of model parameters. 
K=10_240

#currently hacky-- if the combined size of example in train_and_test_loader = ABC then:
traker = TRAKer(model=model, task='pNTK', train_set_size=Ndata,projection_dim=K)
```

### Compute Jacobians of neural network model

```python

traker.load_checkpoint(checkpoint, model_id=0)
for batch in train_and_test_loader:
  # batch should be a tuple of inputs and labels
  traker.featurize(batch=batch, ...)
#this saved a memmap object of the projected gradients to disk.
```

### Compute NTK via Jacobian Contraction

```python

A = torch.from_numpy(np.load('./path/to/0/grads.memmap')).cuda()
NTK = torch.matmul(A,A.T) #NTK has dimensions Ndata x Ndata
```


## Examples
You can find several end-to-end examples in the `examples/` directory.

## Citation
If you use the capabilities developed to compute approximate NTK, consider citing our work!
```
@misc{engel2023robust,
      title={Robust Explanations for Deep Neural Networks via Pseudo Neural Tangent Kernel Surrogate Models}, 
      author={Andrew Engel and Zhichao Wang and Natalie S. Frank and Ioana Dumitriu and Sutanay Choudhury and Anand Sarwate and Tony Chiang},
      year={2023},
      eprint={2305.14585},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
You should also cite the original TRAK repository:
```
@inproceedings{park2023trak,
  title = {TRAK: Attributing Model Behavior at Scale},
  author = {Sung Min Park and Kristian Georgiev and Andrew Ilyas and Guillaume Leclerc and Aleksander Madry},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2023}
}
```

## Installation

We advise installing TRAK V0.2.1, then clone our repository and install locally.

To install TRAK with `CUDA` kernel for fast gradient projection, follow the installation instructions at
[installation FAQs](https://trak.readthedocs.io/en/latest/install.html). You will need compatible versions
of `gcc` and `CUDA toolkit`. 

```
pip install traker[fast]=0.2.1

git clone <This repo>
cd <./this/repo>
pip install -r ./
```