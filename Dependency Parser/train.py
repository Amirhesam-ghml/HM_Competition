import torch 
import numpy as np

import os
from preparedata import Read_data
from preparedata import feature
from torch.utils.data import Dataset, DataLoader
import modeltorch as net
import pickle
import logging
from tqdm import trange
from tqdm import tqdm
from parse import myparser
import random


logging.basicConfig(level=logging.DEBUG)


path_1 = "train.orig.conll"
sampels_1,dicts_1 = Read_data(path_1)
word2num  = dicts_1[0]
pos2num   = dicts_1[1]
label2num = dicts_1[2]
num2label = dicts_1[3]


dicts = [word2num,pos2num,label2num,num2label]
sampels = sampels_1
mydataset = feature(dicts,sampels)

random.seed(12)
dataloader = DataLoader(mydataset, batch_size=1, shuffle=True)

h_units = 200
input_dim = [len(word2num)+3,len(pos2num )+2,len(label2num )+2]
num_classes = 2*len(label2num)+1
model = net.Onelayer(h_units,input_dim ,num_classes )
model.cuda()
write1 = open("Train.vocab","w") 

optimizer =torch.optim.Adagrad(model.parameters(), lr= 0.01)
loss_fn = net.loss_fn
iterations = 20
totalloss = 10000
for i in range(iterations):

    logging.info("-epoch"+str(i))

    totalloss = 0 
    totalsamples = 0

    with tqdm(total=len(dataloader)) as t:
        for sample in dataloader:

            if sample != ['Noneprojective']:

                confs,Labs,words,actions = sample
                output = model(confs.cuda())

                for i in words:

                    write1.write(i[0]+',')
                write1.write('\t') 
                for i in actions:
                    write1.write(i[0]+', ')
                write1.write('\n')

                Labs = torch.squeeze(Labs).type(torch.LongTensor)
                loss = loss_fn(output,Labs.cuda())
            

                totalloss = totalloss*totalsamples + loss*len(Labs)
                totalsamples = totalsamples + len(Labs)
                totalloss = totalloss/totalsamples
                l2 = 0
                for p in model.parameters():
                    l2 = l2 + (p**2).sum()

                loss = loss + 0.5*(10**(-8)) * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.update()

                
    out = (model,dicts)
    torch.save(out, "modelnew")

    p = myparser()
    p.myparser("modelnew","dev.orig.conll","dev.out")
    dev_real   = open("dev.orig.conll" , "r")
    dev_predct = open("dev.out" , "r")
    os.remove("dev.out")
    lines_dev  = dev_real.readlines()
    lines_pred = dev_predct.readlines()
    total_h = 0
    total_l = 0
    cor_h = 0
    cor_l = 0
    for i in range(len(lines_dev)):
        if len(lines_dev[i].split())==10:
            total_h = total_h + 1
            total_l = total_l + 1
            if lines_dev[i].split()[6]==lines_pred[i].split()[6]:
                cor_h = cor_h+1
            if lines_dev[i].split()[7]==lines_pred[i].split()[7]:
                cor_l = cor_l +1
    

    logging.info("-LAS"+str(cor_l/total_l))
    logging.info("-UAS:"+str(cor_h/total_h))   
    logging.info("-Loss:"+str(totalloss))
                

        
     
        
        
        
        