import numpy as np

from abc import ABC, abstractmethod

from Features import Features
import string 
import numpy as np
from operator import methodcaller

import os
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader

import torch
from omegaconf import OmegaConf
from torch import nn

import numpy as np
# import torchvision

from skimage import io, transform
import json





class Onelayer(nn.Module):
    def __init__(self,h_units,input_dim,num_classes):
        super().__init__()
        

        self.seq1 = nn.Sequential(nn.Linear(input_dim,h_units),
                                     nn.Tanh(),
                                      nn.Linear(h_units,num_classes),)
        
    def forward(
        self,input_data,
    ):
        
        output = self.seq1(input_data)
        
        return output        
        
        
def loss_fn(outputs, labels):


    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    return loss 
        
        
        
