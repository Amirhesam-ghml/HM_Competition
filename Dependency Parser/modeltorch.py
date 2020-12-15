import torch 
import numpy as np
from torch import nn
import pickle

import gensim.downloader

CUDA_LAUNCH_BLOCKING="1"


class Onelayer(nn.Module):
    def __init__(self,h_units,input_dim,num_classes):
        super().__init__()
        
        self.word_embed   = nn.Embedding(input_dim[0], 50) #comment to use pretrained model
#         self.word_embed   = nn.Embedding(3, 50)   #uncomment to use pretrained model
        self.pos_embed    = nn.Embedding(input_dim[1], 50)
        self.label_embed  = nn.Embedding(input_dim[2], 50)
        
        self.seq1 = nn.Sequential(nn.Linear(50*48,h_units),
                                     nn.Tanh(),
                                      nn.Linear(h_units,num_classes),)
#         self.lin1 = nn.Linear(50*48,h_units)        #uncomment to use pretrained model
#         self.lin2 = nn.Linear(h_units,num_classes)  #uncomment to use pretrained model


        self.word_embed.weight.data.uniform_(-0.01, 0.01)
        self.pos_embed .weight.data.uniform_(-0.01, 0.01)
        self.label_embed.weight.data.uniform_(-0.01, 0.01)
#         self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50') #uncomment to use pretrained model

        
    def forward(
        self,input_data, word_dict=[]
    ):

        input_data = torch.squeeze(input_data,axis=0)

        words = self.word_embed(input_data[:,0:18]) #comment to use pretrained model
        
        ''' #uncomment to use pretrained model
#         word_id = input_data[:,0:18]
#         batch = word_id.size()[0]
#         print(word_id[2,:])
        
#         word_vec =torch.zeros(batch,18*50).cuda()
#         for i in range(batch):
# #             print("new")
            
# #             print(word_id[i,:])
#             for ind,wid in  enumerate(word_id[i,:]):
# #                 print(wid)
#                 tmp = word_dict.get(wid) 
#                 if wid==0:
#                     word_vec[i,ind*50:(ind+1)*50] = self.word_embed(torch.LongTensor([0]).cuda())
#                 elif wid == 1:
#                     word_vec[i,ind*50:(ind+1)*50] = self.word_embed(torch.LongTensor([1]).cuda())
#                 else:
#                     if tmp in self.glove_vectors.vocab:
#                         word_vec[i,ind*50:(ind+1)*50] = torch.from_numpy(np.array(self.glove_vectors[tmp])).cuda()
#                     else:
#                         word_vec[i,ind*50:(ind+1)*50] = self.word_embed(torch.LongTensor([2]).cuda())
        
#         words = word_vec
        '''
        
        pos = self.pos_embed (input_data[:,18:36])
        labs = self.label_embed(input_data[:,36:])
      
        
        
        words = words.view(-1,18*50)
        pos = pos.view(-1,18*50)
        labs = labs.view(-1,12*50)

        
        
        alldata = torch.cat((words, pos,labs), 1)

#        Cubic activation to use this one need to comment 20 to 22 and uncomment 23 and 24
#         out1 = self.lin1(alldata)#torch.mm(alldata, self.wc1)
#         out2 = out1**3
#         print(out2.size())
#         output = self.lin2(out2)#torch.mm(out2, self.wc2)

        output = self.seq1(alldata)

       
        
        return output        
        
        
def loss_fn(outputs, labels):


    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    return loss 