import torch 
import numpy as np

from preparedata import Read_data
from preparedata import feature
from torch.utils.data import Dataset, DataLoader
import modeltorch as net
import pickle
import argparse
from myparser import myparser



        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()
    
    p = myparser()
    p.myparser(args.m,args.i,args.o)


