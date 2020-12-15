""" 
    Basic feature extractor
"""
from operator import methodcaller
import string 
import numpy as np
import functools
import copy

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class Features:

    def __init__(self,f_max,unk_add,emb_file,data_file,kind="Classify"):
        with open(data_file) as file:
            data = file.read().splitlines()
        if kind == "Train":
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts, self.labels = map(list, zip(*data_split))
            self.labels = np.array(self.labels)
            self.labelname = np.array(list(set(self.labels)))
            self.tokenized_text = [np.array(tokenize(text)) for text in texts]
            
            self.labeldict = dict(zip(self.labelname, range(0, len(self.labelname))))
            self.labeldict_rev = dict(zip( range(0, len(self.labelname)),self.labelname ))
            self.lables_number = np.array([ self.labeldict.get(k) for k in self.labels])

        else:
            self.tokenized_text = [np.array(tokenize(text)) for text in data]
            self.tokenized_text = np.array(self.tokenized_text,dtype=object)
            
        
        self.f_max = f_max
        
        
        with open(emb_file) as file:
            emb = file.read().splitlines()
        
        data_split = map(methodcaller("rsplit", " "), emb)
        data_split = map(lambda x: [x[0],[np.float64(k) for k in x[1:] if k!='']],data_split)
        words,vectors = map(list, zip(*data_split))
        self.emb_dict = dict(zip(words,vectors))
        
        with open(unk_add) as file:
            unk= file.read()
            
        unk_rm = unk.rsplit(' ')
        unk_rm = np.array([x for x in unk_rm if x != ''][1:],dtype=np.float)
        
        self.final_data =  list(map(functools.partial(self.get_features,mean_vec = unk_rm ), self.tokenized_text)) 
        self.final_data = np.vstack(np.array(self.final_data,dtype=np.float))

    def get_features(self, tokenized,mean_vec):

        F1 = np.zeros((len(tokenized),len(mean_vec)))
        for index,k in enumerate(tokenized):
            dict_value = self.emb_dict.get(k.lower())
            if dict_value == None:
                F1[index,:] = mean_vec
            else:
                F1[index,:] = dict_value

        if F1.shape[0]< self.f_max:
            F1 = np.append(F1,np.zeros((self.f_max-F1.shape[0],F1.shape[1])),axis=0)
        else:
            F1 = F1[0:self.f_max,:]  

        return F1.flatten()
    
    
    
    
    
    
