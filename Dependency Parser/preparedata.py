import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import itertools
from gen_feat import genfeat

def Read_data(path,create_dict=True):
    '''
    This function is used to read inputs like taring.conll or dev.conll line by line and generate all information about each sentence. And in       case of training data set create Necessary dictionaries.        
    '''
    samples = []
    with open(path) as f:
        word, pos, head, label = [], [], [], []
        word_dict  = []
        pos_dict   = []
        label_dict = []
        a2 = []
        a3 = []
        a5 = []
        a8 = []
        a9 = []
        real_word = []
        for line in f.readlines():

            sp = line.split('\t')
    #         print(len(sp))
            if len(sp) == 10:
                if '-' not in sp[0]:
                
                    word_dict.append(sp[1].lower() )
                    pos_dict.append(sp[4])
                    label_dict.append(sp[7])
                    
                    
                    a2.append(sp[2])
                    a3.append(sp[3])
                    a5.append(sp[5])
                    a8.append(sp[8])
                    a9.append(sp[9])
                    word.append(sp[1].lower() )
                    real_word.append(sp[1] )
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                samples.append({'word': word, 'pos': pos, 'head': head, 'label': label,'a2':a2,'a3':a3,'a5':a5,'a8':a8,'a9':a9,'real_word': real_word})
                word, pos, head, label,a2,a3,a5,a8,a9,real_word = [], [], [], [],[],[],[],[],[],[]
        if len(word) > 0:
            samples.append({'word': word, 'pos': pos, 'head': head, 'label': label,'a2':a2,'a3':a3,'a5':a5,'a8':a8,'a9':a9,'real_word': real_word})
        
        if create_dict:
            word_dict   = list(set(word_dict))
            pos_dict    = list(set(pos_dict))
            label_dict  = list(set(label_dict))
            word2num    = dict(zip(word_dict, range(3, len(word_dict)+3)))
            word2num['ROOT'] = 0
            pos2num     = dict(zip(pos_dict, range(2, len(pos_dict)+2)))
            label2num   = dict(zip(label_dict, range(2, len(label_dict)+2)))
            num2label   = dict(zip(range(2, len(label_dict)+2),label_dict))
            
            
            dicts = [word2num,pos2num,label2num,num2label]
            
            return samples,dicts
        else:
            return samples
        
        
class feature(Dataset):
    '''
    This class is an expansion of pytorch Dataset class, to lazily load inputs for training
    '''
    def __init__(self,dicts,samples):
        
        self.word2num  = dicts[0]
        self.pos2num   = dicts[1]
        self.label2num = dicts[2]
        self.num2label = dicts[3]
        self.samples   = samples
        
    def __len__(self): 
        
        return len(self.samples)
    

    
    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        
        words = sample['word']
        pos   = sample['pos']
        label = sample['label']
        heads  = sample['head']
        
        actions = []
        
        lensen = len(words)
        stack  = ['ROOT']
        buffer = list(range(1,lensen+1 ))
        

        confs  = np.zeros((2*lensen,48),dtype = np.int)
        Labs   = np.zeros((2*lensen))
        L = len(self.label2num)
        

        for ind,h in enumerate(heads):

            if h!=0:
                Min = min(ind+1,heads[ind])
                Max = max(ind+1,heads[ind])
            
                for j in range(Min+1,Max):

                    headstmp = np.array(heads)

                    tmp = np.where(headstmp==j)[0]+1

                    tmp2 = np.where(np.logical_or(tmp<Min,tmp>Max))[0]

                    if (heads[j-1]>Max or heads[j-1]<Min or heads[j-1] ==0 or len(tmp2)>0):

                        return "Noneprojective"

                
        count = 0
        dicts = [self.word2num,self.pos2num,self.label2num,self.num2label]
        class_feat = genfeat(dicts)
        heads2 = -1*np.ones(len(words),dtype=np.int)
        lable_val = -1*np.ones(len(words),dtype=np.int)
        while (stack!=['ROOT'] or buffer !=[] ):
            

            
            
            confs [count,:] = class_feat.gen(stack,buffer,heads2,words,pos,lable_val)
            
            #create labels

            if len(stack)==1:

                Labs[count] = 0
                stack = stack + [buffer.pop(0)]
                actions.append('Shift')
            elif len(stack)==2:

                if (heads[stack[-1]-1]==0 and len(buffer)==0):
                    #right_action
                    Labs[count] = L + self.label2num[label[stack[-1]-1]]-1
                    actions.append('Right')
                    heads2[stack[-1]-1]=0
                    lable_val[stack[-1]-1] = self.label2num[label[stack[-1]-1]]
                    del stack[-1]
                else:
                    Labs[count] = 0 
                    stack = stack + [buffer.pop(0)]
                    actions.append('Shift')
            else:

                #left_action
                if heads[stack[-2]-1] == stack[-1]:
                    Labs[count] = self.label2num[label[stack[-2]-1]]-1
                    

                    actions.append('Left')

                    heads2[stack[-2]-1] = stack[-1]
                    lable_val[stack[-2]-1] = self.label2num[label[stack[-2]-1]]
                    del stack[-2]
                #right_action
                elif (heads[stack[-1]-1] == stack[-2]) and (  len(np.where( np.array( [heads[i-1] for i in buffer])== stack[-1])[0])==0 ):
                    Labs[count] = L + self.label2num[label[stack[-1]-1]]-1

                    actions.append('Right')
                    heads2[stack[-1]-1] = stack[-2]
                    lable_val[stack[-1]-1] = self.label2num[label[stack[-1]-1]]
                    del stack[-1]
                else:
                    Labs[count] = 0
                    stack = stack + [buffer.pop(0)]

                    actions.append('Shift')

            count = count+1
        return confs,Labs,words,actions
    
                
                
 

        
        
        
        
        
        
     