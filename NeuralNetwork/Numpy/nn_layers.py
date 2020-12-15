import numpy as np

from abc import ABC, abstractmethod


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Features import Features
import string 
import numpy as np
from operator import methodcaller


class NNComp(ABC):
    
    def __init__(self,hid_size,lr,batches,epochs,emb_size,label_size):
        
        self.hid_size   = hid_size
        self.lr         = lr
        self.batches    = batches
        self.epochs     = epochs
        self.emb_size   = emb_size
        self.label_size = label_size
        


    def initilization(self):
        x_dim = self.emb_size
        w1    = np.random.randn(x_dim,self.hid_size)*0.01
        b1    = np.zeros((1,self.hid_size))
        w2    = np.random.randn(self.hid_size,self.label_size)*0.01
        b2    = np.zeros((1,self.label_size))

        parameters = {'W1':w1,'b1':b1,'W2':w2,'b2':b2}
        return parameters

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def relu(self,X):
        return np.maximum(X,0)

    def forward_onestep(self,X,w,b,activation):

        if activation == 'sigmoid':
            out1 = np.matmul(X,w)+b
            out2 = self.sigmoid(out1)
        elif activation == 'relu':
            out1 = np.matmul(X,w)+b
            out2 = self.relu(out1)
        elif activation == 'softmax':
            out1 = np.matmul(X,w)+b
            tmp = (out1.T-np.amax(out1,axis=1) ).T
            out2 = np.exp( (out1.T-np.amax(out1,axis=1) ).T )
            out2 = (out2.T/np.sum(out2,axis = 1)).T
        return out2,(X,w,b,out1)

    def forward(self,X,parameters):
        L = len(parameters) //2
        A_prev = X
        caches = []
        for i in range(1,L):
            A_prev,cache =self.forward_onestep(A_prev,parameters['W'+str(i)],
                                           parameters['b'+str(i)],'relu')
            caches.append(cache)

        A_prev,cache = self.forward_onestep(A_prev,parameters['W'+str(2)],
                                       parameters['b'+str(2)],'softmax')
        caches.append(cache)
        return A_prev,caches

    def cost(self,AZ,Y,parameters):
        RegCost = 0
        for l in range(1,3):
            RegCost = RegCost + 0.05*np.sum( (parameters["W" + str(l)].flatten())**2)

        Mask = np.zeros(AZ.shape)

        Mask[np.arange(AZ.shape[0]),np.array(Y)]=1
        Cost = Mask*AZ

        rows, cols = np.nonzero(Cost) 
        Cost = Cost[rows, cols]
        Cost = -np.log(Cost)
        Cost = Cost.flatten()
        return np.sum(Cost)/len(Y)


    def linearBackProp(self,dZ,cache):
        A_prev, W, b = cache

        dW = np.matmul(np.transpose(A_prev),dZ) 
        dA = np.matmul(dZ,np.transpose(W))
        db = np.sum(dZ,axis=0,keepdims=True)
        return dA, dW, db

    def reluBackProp(self,dA,Z):
    #         dZ = dA.where(dA>0,dA,1)
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ



    def backward(self,AZ,Y,caches):
        Mask = np.zeros(AZ.shape)
        Mask[np.arange(AZ.shape[0]),np.array(Y)]=1
        dZ2 = (AZ - Mask)/len(Y)

        dA1,dW2,db2 = self.linearBackProp(dZ2,caches[1][0:3])
        dZ1 = self.reluBackProp(dA1,caches[0][3])


        dA0,dW1,db1 = self.linearBackProp(dZ1,caches[0][0:3])
        grads = {}
        grads["dA0" ] = dA0
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dA1" ] = dA1
        grads["dW2"] = dW2
        grads["db2"] = db2
        return grads

    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2 

    # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*(grads["dW" + str(l+1)]+0*parameters["W" + str(l+1)])
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        return parameters






    def run(self,xtrain,ytrain,print_cost=True):#lr was 0.009

        costs = []                         # keep track of cost

        learning_rate  = self.lr
        num_iterations = self.epochs
        
        parameters = self.initilization()
        batch = self.batches
        
        Numberofdata = xtrain.shape[0]
        xTrain = xtrain[0:int(np.floor(0.8*Numberofdata)),:]
        yTrain = ytrain[0:int(np.floor(0.8*Numberofdata))]
        
        xTest = xtrain[int(np.ceil(0.8*Numberofdata)):,:]
        yTest = ytrain[int(np.ceil(0.8*Numberofdata)):]
        


        Train_cost = []
        Test_cost = []
        for i in range(1, num_iterations+1):
            

            for j in range(numberofbatch):
                if j!= numberofbatch-1:
                    xtrainSUB = xTrain[j*batch:(j+1)*batch,:]
                    ytrainSUB = yTrain[j*batch:(j+1)*batch]
                else:
                    xtrainSUB = xTrain[j*batch:,:]
                    ytrainSUB = yTrain[j*batch:]
                # Forward propagation: 

                AZ3, caches = self.forward(xtrainSUB,parameters)

                # Compute cost.
                mycost = self.cost(AZ3,ytrainSUB,parameters)

                # Backward propagation.
                grads = self.backward(AZ3,ytrainSUB,caches)
                # Update parameters.
                parameters = self.update_parameters(parameters, grads, learning_rate)
            costs.append(mycost)
            Train_cost.append(mycost)
            print ("Train_Cost after iteration %i: %f" %(i, mycost))
            print ("Train Accuracy after iteration %i: " %(i))
            probs,_ = self.forward(xTrain,parameters)
            labels = np.argmax(probs,axis=1) 
            nonzero = np.count_nonzero(labels -yTrain)
            accu = (len(yTrain) - nonzero)/len(yTrain)
            print(accu)
            
            
            probs,_ = self.forward(xTest,parameters)
            mycost = self.cost(probs,yTest,parameters)
            Test_cost.append(mycost)
            print ("Test_Cost after iteration %i: %f" %(i, mycost))
            print ("Test Accuracy after iteration %i: " %(i))
            
            labels = np.argmax(probs,axis=1) 
            nonzero = np.count_nonzero(labels -yTest)
            accu = (len(yTest) - nonzero)/len(yTest)
            print(accu)
            

        return parameters,Train_cost,Test_cost

        
        
        
        
        
        
        
