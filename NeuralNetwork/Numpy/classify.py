import pickle
import argparse
from nn_layers import NNComp
from Features import Features

def classify(args):
    out = pickle.load(open(args.m, 'rb'))
    params = out[0]
    dict_rev = out[1]
    if args.m == "odia" or "odia.torch":
        mydata    = Features(350,'unk-odia.vec',"fasttext.wiki.300d.vec",args.i)       
    else:
        mydata    = Features(350,'unk.vec',"glove.6B.50d.txt",args.i)
        
    model = NNComp(20,0.01,32,40,15000,4)
    out,_ = model.forward(mydata.final_data,params)
    
    labels = np.argmax(out,axis=1) 
    
    preds = np.array([ dict_rev.get(str(k)) for k in labels])
    
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
