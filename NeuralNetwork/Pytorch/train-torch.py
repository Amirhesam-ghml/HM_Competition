import pickle
import argparse
import nn_layers_torch as net
from Features import Features
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# import torchvision

from skimage import io, transform
import json



def train(args):
    if args.i == "datasets/odia.train.txt":
        mydata    = Features(args.f,'unk-odia.vec',args.E,args.i,"Train")
        dim = 300
    else:
        mydata    = Features(args.f,'unk.vec',args.E,args.i,"Train")
        dim = 50
        
    model = net.Onelayer(args.u,args.f*dim,len(mydata.labelname) )
    
    Numberofdata = mydata.final_data.shape[0]
    

    xtrain = mydata.final_data[0:int(np.floor(0.8*Numberofdata)),:]
    ytrain = mydata.lables_number[0:int(np.floor(0.8*Numberofdata))]
    
    xvald = mydata.final_data[int(np.ceil(0.8*Numberofdata)):,:]
    yvald = mydata.lables_number[int(np.ceil(0.8*Numberofdata)):]
    print(xvald.shape)

    batch = args.b
    
    numberofbatch  = int(np.ceil(xtrain.shape[0]/batch))
    num_iterations = args.e

    optimizer =torch.optim.Adam(model.parameters(), lr= args.l, eps=1.0e-08,weight_decay=0.01)
    loss_fn = net.loss_fn

    train_losses =[]
    vald_losses =[]
    for i in range(1, num_iterations+1):
        loss_avg = 0
        model.train()
        for j in range(numberofbatch):
            if j!= numberofbatch-1:
                xtrainSUB = xtrain[j*batch:(j+1)*batch,:]
                ytrainSUB = ytrain[j*batch:(j+1)*batch]
            else:
                xtrainSUB = xtrain[j*batch:,:]
                ytrainSUB = ytrain[j*batch:]
            xtrainSUB = torch.from_numpy(xtrainSUB).float()
            ytrainSUB = torch.from_numpy(ytrainSUB).long()
            output = model(xtrainSUB)
            loss = loss_fn(output,ytrainSUB)
            loss_avg = loss_avg + loss*ytrainSUB.shape[0] 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print ("Train_Cost after iteration %i: %f" %(i, loss_avg/ytrain.shape[0]))
        train_losses.append(loss_avg/ytrain.shape[0])
        model.eval()
        with torch.no_grad():
            print ("Train Accuracy after iteration %i: " %(i))
            probs = model.forward(torch.from_numpy(xtrain).float())
            labels = np.argmax(probs,axis=1) 
            nonzero = np.count_nonzero(labels -ytrain )
            accu = (len(ytrain) - nonzero)/len(ytrain )
            print(accu)
            
       
            
            probs = model.forward(torch.from_numpy(xvald).float())
            labels = np.argmax(probs,axis=1) 
            lossvald = loss_fn(probs, torch.from_numpy(yvald).long())
            vald_losses.append(lossvald)
            print ("Vald_Cost after iteration %i: %f" %(i, lossvald))
            print ("Validation Accuracy after iteration %i: " %(i))
            
            nonzero = np.count_nonzero(labels -yvald )
            accu = (len(yvald) - nonzero)/len(yvald )
            print(accu)
            
    plt.plot(train_losses,'b',vald_losses,'r--')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.suptitle('Train/vald loss')
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Train')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()       
            
    out = (model,args.u,args.f,dim,len(mydata.labelname),mydata.labeldict_rev)

    
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')

    args = parser.parse_args()

    train(args)
