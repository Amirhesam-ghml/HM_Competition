import pickle
import argparse
from nn_layers import NNComp
from Features import Features
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def train(args):
    if args.i == "datasets/odia.train.txt":
        mydata    = Features(args.f,'unk-odia.vec',args.E,args.i,"Train")
        dim = 300
    else:
        mydata    = Features(args.f,'unk.vec',args.E,args.i,"Train")
        dim = 50
    model = NNComp(args.u,args.l,args.b,args.e,args.f*dim,len(mydata.labelname) )

    param,Train_cost,Test_cost = model.run(mydata.final_data,mydata.lables_number)
    
    plt.plot(Train_cost,'b',Test_cost,'r--')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
#     plt.xticks([0,5,10,15,20,25])
    plt.suptitle('Train/vald loss')
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Train')
    plt.legend(handles=[red_patch,blue_patch])
    
    plt.show()   
    
    
    out = (param,mydata.labeldict_rev)
#     pickle.dump(out,open(args.o, 'wb') )

    
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
