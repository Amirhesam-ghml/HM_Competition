import numpy as np

class genfeat():
    
    '''
    This is the main class used to generate feature. We use this class to generate features for both training and parsing part
    This class contains two main function: 
    1."ch": to find childeren of a word based on given heads 
    2."gen": a function that given (stack, buffer and heads) and also the information about the words, meaning actual words, POS and label, it       outputs the vector of sie 48 which     is the generated feature from the given configuration.
    '''
    def __init__(self,dicts):
        
        self.word2num  = dicts[0]
        self.pos2num   = dicts[1]
        self.label2num = dicts[2]
        self.num2label = dicts[3]
    
    #This function is used to find left and right children of an a word(shown by ind) using heads(heads indentified so far)
    def ch(self,heads,ind):

        if ind=="ROOT":
            ind =0
       
        heads = np.array(heads)
        childs = np.where(heads == ind)[0]+1
        childs = np.array(childs)
        childs_left  = childs[childs<ind]
        childs_right = childs[childs>ind]

        return childs_left,childs_right
    
    #This is the main function used to find all 4 categories explained in the report 
    def gen(self,stack, buffer,heads,words,pos,label):
        lensen = len(words)
        
        
        #first 3 words of stack
        
        cat1= []
        cat1_pos = []
        if len(stack)==1:
            cat1 = cat1 + [0]+[1]+[1]
            cat1_pos = cat1_pos + [0] + [0] + [0]
        elif  len(stack)==2:
            tmp_w = self.word2num.get(words[stack[-1]-1].lower())
            if tmp_w==None:
                tmp_w =2
            cat1 = cat1 + [tmp_w] +[0]+[1]
            tmp_p = self.pos2num.get(pos[stack[-1]-1])
            if tmp_p==None:
                tmp_p=1
            cat1_pos = cat1_pos + [tmp_p] +[0]+[0]

        else:                
            for i in range(1,4):
                if stack[-i]=='ROOT':
                    cat1 = cat1 + [0]
                    cat1_pos = cat1_pos + [0]

                else:
                    tmp_w = self.word2num.get(words[stack[-i]-1].lower())
                    tmp_pos = self.pos2num.get(pos[stack[-i]-1])
                    if tmp_w ==None:
                        tmp_w=2
                    if tmp_pos ==None:
                        tmp_pos=1
                    cat1 = cat1 + [tmp_w]
                    cat1_pos = cat1_pos + [tmp_pos]


        #first 3 words of buffer
        cat2= []
        cat2_pos = []
        cat2_lab = []
        if len(buffer)==0:
            cat2 = [1,1,1]
            cat2_pos = [0,0,0]
            cat2_lab = [0,0,0]

        elif len(buffer)==1:
            tmp_w = self.word2num.get(words[buffer[0]-1].lower())
            tmp_pos = self.pos2num.get(pos[buffer[0]-1])
            if tmp_w==None:
                 tmp_w=2
            if tmp_pos==None:
                 tmp_w=1
            cat2 = cat2 + [ tmp_w]+[1]+[1]
            cat2_pos = cat2_pos + [ tmp_pos ]+[0]+[0]

        elif len(buffer)==2:
            tmp_w_1 = self.word2num.get(words[buffer[0]-1].lower())
            tmp_w_2 = self.word2num.get(words[buffer[1]-1].lower())
            tmp_pos_1 = self.pos2num.get(pos[buffer[0]-1])
            tmp_pos_2 = self.pos2num.get(pos[buffer[1]-1])
            if tmp_pos_1 ==None:
                tmp_pos_1 =1
            if tmp_pos_2 ==None:
                tmp_pos_2 =1
            if tmp_w_1 ==None:
                tmp_w_1=2
            if tmp_w_2 ==None:
                tmp_w_2=2
            cat2 = cat2 + [tmp_w_1]+  [ tmp_w_2]+[1]
            cat2_pos = cat2_pos + [tmp_pos_1]+ [ tmp_pos_2]+[0]

        else:
            tmp_w_1 = self.word2num.get(words[buffer[0]-1].lower())
            tmp_w_2 = self.word2num.get(words[buffer[1]-1].lower())
            tmp_w_3 = self.word2num.get(words[buffer[2]-1].lower())
            tmp_pos_1 = self.pos2num.get(pos[buffer[0]-1])
            tmp_pos_2 = self.pos2num.get(pos[buffer[1]-1])
            tmp_pos_3 = self.pos2num.get(pos[buffer[2]-1])
            if tmp_pos_1 ==None:
                tmp_pos_1 =1
            if tmp_pos_2 ==None:
                tmp_pos_2 =1
            if tmp_pos_3 ==None:
                tmp_pos_3 =1
            if tmp_w_1 ==None:
                tmp_w_1=2
            if tmp_w_2 ==None:
                tmp_w_2=2
            if tmp_w_3 ==None:
                tmp_w_3=2
            cat2 = cat2 + [tmp_w_1 ]+[tmp_w_2 ]+ [tmp_w_3 ]
            cat2_pos = cat2_pos + [tmp_pos_1]+ [tmp_pos_2 ]+ [tmp_pos_3 ]


       # first and second leftmost/ rightmost
        cat3 = []
        cat3_pos = []
        cat3_lab = []
        if len(stack)<2:
            left_childs_1,right_childs_1 = self.ch(heads,0) 
        else:

            left_childs_1,right_childs_1 =self.ch(heads,stack[-1])
            left_childs_2,right_childs_2 = self.ch(heads,stack[-2])


        left_eff_len = min(len(left_childs_1),2) 
        right_eff_len = min(len(right_childs_1),2) 

        for i in range(left_eff_len):
            tmp_w = self.word2num.get(words[left_childs_1[i]-1].lower())
            tmp_pos = self.pos2num.get(pos[left_childs_1[i]-1])
            if tmp_w ==None:
                tmp_w =2
            if tmp_pos==None:
                tmp_pos=1
            cat3 = cat3 + [ tmp_w ]
            cat3_pos = cat3_pos + [ tmp_pos]
            if label[left_childs_1[i]-1]!=-1:
                cat3_lab = cat3_lab + [label[left_childs_1[i]-1]]

        cat3 = cat3 + [1]*(2-len(cat3))

        cat3_pos = cat3_pos + [0]*(2-len(cat3_pos))
        cat3_lab = cat3_lab + [0]*(2-len(cat3_lab))

        for i in range(1,right_eff_len+1):
            tmp_w = self.word2num.get(words[right_childs_1[-i]-1].lower())
            tmp_pos = self.pos2num.get(pos[right_childs_1[-i]-1])
            if tmp_w ==None:
                tmp_w =2
            if tmp_pos==None:
                tmp_pos=1
            cat3 = cat3 + [tmp_w]
            cat3_pos = cat3_pos + [tmp_pos]
            if label[right_childs_1[-i]-1]!=-1:
                cat3_lab = cat3_lab + [label[right_childs_1[-i]-1]]


        cat3 = cat3 + [1]*(4-len(cat3))
        cat3_pos = cat3_pos + [0]*(4-len(cat3_pos))
        cat3_lab = cat3_lab + [0]*(4-len(cat3_lab))     

        if  len(stack)<2:

            cat3 = cat3 + [1]*(8-len(cat3))
            cat3_pos = cat3_pos + [0]*(8-len(cat3_pos))
            cat3_lab = cat3_lab + [0]*(8-len(cat3_lab)) 
        else:
            left_eff_len = min(len(left_childs_2),2) 
            right_eff_len = min(len(right_childs_2),2)
            for i in range(left_eff_len):
                tmp_w = self.word2num.get(words[left_childs_2[i]-1].lower())
                tmp_pos = self.pos2num.get(pos[left_childs_2[i]-1])
                if tmp_w ==None:
                    tmp_w =2
                if tmp_pos==None:
                    tmp_pos=1
                cat3 = cat3 + [tmp_w]
                cat3_pos = cat3_pos + [tmp_pos]
                if label[left_childs_2[i]-1]!=-1:
                    cat3_lab = cat3_lab + [label[left_childs_2[i]-1]]


            cat3 = cat3 + [1]*(6-len(cat3))
            cat3_pos = cat3_pos + [0]*(6-len(cat3_pos))
            cat3_lab = cat3_lab + [0]*(6-len(cat3_lab))

            for i in range(1,right_eff_len+1):
                tmp_w = self.word2num.get(words[right_childs_2[-i]-1].lower())
                tmp_pos = self.pos2num.get(pos[right_childs_2[-i]-1])
                if tmp_w ==None:
                    tmp_w =2
                if tmp_pos==None:
                    tmp_pos=1
                cat3 = cat3 + [tmp_w]
                cat3_pos = cat3_pos + [tmp_pos]
                if label[right_childs_2[-i]-1]!=-1:
                    cat3_lab = cat3_lab + [label[right_childs_2[-i]-1]]


            cat3 = cat3 + [1]*(8-len(cat3))
            cat3_pos = cat3_pos + [0]*(8-len(cat3_pos))
            cat3_lab = cat3_lab + [0]*(8-len(cat3_lab))    

        # leftmost of the leftmost/ rightmost of the rightmost
        cat4 = []
        cat4_pos = []
        cat4_lab = []

        #1
        if len(stack)==1: 
            left_childs_1,right_childs_1 = self.ch(heads,0)
        else:
            left_childs_1,right_childs_1 = self.ch(heads,stack[-1])
        #left_1
        if len(left_childs_1)>0:
            left_left, _  = self.ch(heads,left_childs_1[0])

        if (len(left_childs_1)==0 or len(left_left)==0):
            cat4 = cat4 + [1]
            cat4_pos = cat4_pos + [0]
            cat4_lab = cat4_lab + [0]
        else:
            tmp_w = self.word2num.get(words[left_left[0]-1].lower())
            tmp_pos = self.pos2num.get(pos[left_left[0]-1])
            if tmp_w ==None:
                tmp_w =2
            if tmp_pos==None:
                tmp_pos=1
            cat4     = cat4 + [tmp_w ]
            cat4_pos = cat4_pos + [ tmp_pos]
            if label[left_left[0]-1]!=-1:
                cat4_lab = cat4_lab   + [ label[left_left[0]-1]]
            else: 
                 cat4_lab = cat4_lab   + [0]
        #right_1
        if len(right_childs_1)>0:
            _ ,right_right = self.ch(heads,right_childs_1[-1])

            
        if (len(right_childs_1)==0 or len(right_right)==0):
        #                 print("here")
            cat4 = cat4 + [1]
            cat4_pos = cat4_pos + [0]
            cat4_lab = cat4_lab + [0]
        else:

            tmp_w = self.word2num.get(words[right_right[-1]-1].lower())
            tmp_pos = self.pos2num.get(pos[right_right[-1]-1])
            if tmp_w ==None:
                tmp_w =2
            if tmp_pos==None:
                tmp_pos=1
            cat4     = cat4 + [tmp_w ]
            cat4_pos = cat4_pos +  [tmp_pos]
            if label[right_right[-1]-1]!=-1:
                cat4_lab = cat4_lab   + [ label[right_right[-1]-1]] 
            else:
                cat4_lab = cat4_lab   + [0]
        #2
        if len(stack)==1: 
            cat4 = cat4 + [1,1] 
            cat4_pos = cat4_pos + [0,0]
            cat4_lab = cat4_lab + [0,0]
        else:

            #left_2
            if len(left_childs_2)>0:
                left_left,_ = self.ch(heads,left_childs_2[0])

            if (len(left_childs_2)==0 or len(left_left)==0):
                cat4 = cat4 + [1]
                cat4_pos = cat4_pos + [0]
                cat4_lab = cat4_lab + [0]
            else:
                tmp_w = self.word2num.get(words[left_left[0]-1].lower())
                tmp_pos = self.pos2num.get(pos[left_left[0]-1])
                if tmp_w ==None:
                    tmp_w =2
                if tmp_pos==None:
                    tmp_pos=1
                cat4     = cat4 + [ tmp_w ]
                cat4_pos = cat4_pos + [ tmp_pos ]
                if label[left_left[0]-1]!=-1:
                    cat4_lab = cat4_lab   +  [ label[left_left[0]-1]] 
                else: 
                    cat4_lab = cat4_lab   + [0]
            #right_2
            if len(right_childs_2)>0:
                _,right_right = self.ch(heads,right_childs_2[-1])
            if (len(right_childs_2)==0 or len(right_right)==0):
                cat4 = cat4 + [1]
                cat4_pos = cat4_pos + [0]
                cat4_lab = cat4_lab + [0]
            else:
                tmp_w = self.word2num.get(words[right_right[-1]-1].lower())
                tmp_pos = self.pos2num.get(pos[right_right[-1]-1])
                if tmp_w ==None:
                    tmp_w =2
                if tmp_pos==None:
                    tmp_pos=1
                cat4     = cat4 + [ tmp_w ]
                cat4_pos = cat4_pos + [ tmp_pos]
                if label[right_right[-1]-1]!=-1:
                    cat4_lab = cat4_lab   + [ label[right_right[-1]-1]] 
                else:
                    cat4_lab = cat4_lab   + [0]



        out = cat1 + cat2+ cat3+ cat4 + cat1_pos + cat2_pos  + cat3_pos + cat4_pos + cat3_lab  + cat4_lab
        out = np.array(out,dtype=np.int)
        

        return out