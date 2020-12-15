import pickle
import argparse
# from nn_layers import NNComp
from Features import Features

import pickle
import argparse
import nn_layers_torch as net
from Features import Features
import numpy as np
import torch


import numpy as np

from abc import ABC, abstractmethod

from Features import Features
import string 
import numpy as np
from operator import methodcaller

import os
from copy import deepcopy
import utils
from torch.utils.data import Dataset, DataLoader

import torch
from omegaconf import OmegaConf
from torch import nn

import numpy as np

from skimage import io, transform
import json


def classify(args):
    out = pickle.load(open(args.m, 'rb'))
    dict_rev = out[5]
    model = out[0]
    if args.m == ("odia" or "odia.torch"):
        mydata    = Features(out[2],'unk-odia.vec',"fasttext.wiki.300d.vec",args.i)     
        
    else:
        mydata    = Features(out[2],'unk.vec',"glove.6B.50d.txt",args.i)
        
    
    model.eval()
    
    
    out= model.forward(torch.from_numpy(mydata.final_data).float())
    
    labels = np.argmax(out.detach().numpy(),axis=1) 
    
    preds = np.array([ dict_rev.get(k) for k in labels])
    

    with open(args.o, "w") as file:
        for pred in preds:
            file.write(pred+"\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()

    classify(args)
