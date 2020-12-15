
import torch 
import numpy as np

from preparedata import Read_data
from preparedata import feature
from torch.utils.data import Dataset, DataLoader
import modeltorch as net
import pickle
import argparse
from gen_feat import genfeat


class myparser():

        '''
        The class used to parse any inputs
        '''
        def myparser(self,model,input_data,outname):

                #reading the input 
                samples = Read_data(input_data,create_dict=False)

                #loading trained model
                out = torch.load(model,map_location=torch.device('cpu'))

                model = out[0]
                dicts = out[1]

                # Creating trained dictionaries
                word2num  = dicts[0]
                pos2num   = dicts[1]
                label2num = dicts[2]
                num2label = dicts[3]
                num2word = dicts[4]
                num2pos = dicts[5]
                
                #creating a gen_feat class
                class_feat = genfeat(dicts)
                
                L = len(num2label)
                
                
                f = open(outname, 'a')
                #write1 = open("DEV.vocab","w")
                C = 1
                for sample in samples:
                    
                    
                    #write1.write("sample" + str(C)+":"+"\n")
                    C=C+1
                    pred_label = []
                    words = sample['word']
                    pos   = sample['pos']
                    
                    
                    #Initialize heads, labels, stack and buffer for each sample 
                    heads = -1*np.ones(len(words),dtype=np.int)
                    lable_val = -1*np.ones(len(words),dtype=np.int)
                    actions = []
                    lensen = len(words)
                    stack  = ['ROOT']
                    buffer = list(range(1,lensen+1 ))




                    count = 0
                     
                    while count < 2*lensen:
                        
                        #generating features using gen_feat class functions
                        confs = class_feat.gen(stack,buffer,heads,words,pos,lable_val)
                        confs = np.expand_dims(confs,axis=0)
                        confs = np.expand_dims(confs,axis=0)
                        confs = torch.from_numpy(confs)

                        model.eval()

                        with torch.no_grad():
                            out_data = model(confs,dicts[4])

 
                        label = torch.argmax(out_data).item()

                        if (buffer==[] and label==0):
                            label = torch.argsort(out_data,axis=1)[0][-2].item()
            
                        

                        if len(stack)==1:
                            label = 0
                        if label == 0:
                            pred_label.append("Shift")
                            if buffer!=[]:
                                stack = stack + [buffer.pop(0)]
                        elif label > L:
                            pred_label.append("Right")
                            if stack[-2]=="ROOT":
                                heads[stack[-1]-1] =  0                    
                            else:
                                heads[stack[-1]-1] =  stack[-2]

                            lable_val[stack[-1]-1] = label -L
                            del stack[-1]

                        else :
                            pred_label.append("Left")
                            if stack[-2]=="ROOT":

                                pass
                            else:
                                heads[stack[-2]-1] =  stack[-1]
                                lable_val[stack[-2]-1] = label
                                del stack[-2]

                        
                        count = count+1 


                    lable_name = np.array([num2label.get(i+1) for i in lable_val])
                    
                    #Writing the final CONLL file
                    for i in range(len(words)):

                        if heads[i] == -1:
                            Head = str(0)
                        else:
                            Head = str(heads[i])#str(sample["head"][i])

                        if lable_name[i] == None:
                            Lname = '-'
                        else:
                            Lname = lable_name[i]


                        f.write(str(i+1)+'\t'+str(sample['real_word'][i])+'\t'+str(sample["a2"][i])+'\t'+str(sample["a3"][i])+'\t'+str(sample["pos"][i])+'\t'+str(sample["a5"][i])+'\t'+Head+'\t'+Lname+'\t'+str(sample["a8"][i])+'\t'+str(sample["a9"][i]) )                


                    f.write('\n')


                f.close()

         