"""Train the model"""

import argparse
import logging
import os
import os.path as osp
from HatefulMemesD import  HatefulMemesDataset
import copy

import torchvision
import numpy as np
import torch 
from torch import nn
import torch.optim as optim
from tqdm import trange
from tqdm import tqdm

import utils
import Model.visual_bert as net
# from model.data_loader import DataLoader
from evaluate import evaluate

import pickle
import json
from transformers.modeling_bert import (
    BertConfig,
    BertEncoder,
    BertForPreTraining,
    BertLayer,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/hesam/owncode',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='/home/hesam/owncode',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer_1,optimizer_2, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    h_accu = 0
    h_size = 0

    with tqdm(total=len(dataloader)) as t:

                
#         model.odr.requires_grad =  False
        for train_batch in dataloader:
            # fetch the next training batch
            labels_batch = train_batch['label'].cuda()
            for k in train_batch:
                if(hasattr(train_batch[k],'cuda' )):
                    train_batch[k] = train_batch[k].cuda()

            output_batch = model(train_batch)
            scores = output_batch

            wieghts_list = [model.wc1,model.wc2,model.wc3,model.wc4]
            loss = loss_fn(scores, labels_batch,wieghts_list)
#             print(loss)
            loss_neg = -loss_fn(scores, labels_batch,wieghts_list)
            model.odr.requires_grad =  True
#             print()
            myeps = 0.1
            for index_1 in range(10):
                
                optimizer_2.zero_grad()
                if index_1==0:
                    loss_neg.backward(retain_graph=True)
                optimizer_2.step()

                temp = copy.deepcopy(model.odr)
                temp = temp.detach()
                new_params = myeps*temp/torch.norm(temp.float())#torch.min(temp ,myeps*(torch.ones(model.odr.size()).cuda() ))

                if torch.norm(temp.float())>myeps:
                    with torch.no_grad():
                         model.odr.copy_(new_params)
#                 print(model.odr)
            model.odr.requires_grad =  False
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer_1.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer_1.step()

            out_labels_soft = nn.Softmax(dim=1)(scores)
            out_labels = out_labels_soft[:,0]<0.5
            out_labels = out_labels.detach().cpu().numpy().astype('int')

            labels_batch = labels_batch.cpu().numpy()
            h_accu = h_accu + np.sum(out_labels == labels_batch)
            h_size = h_size + len(out_labels)


            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    logging.info("- Train metrics: " + "h-accu:"+str(h_accu/h_size)+ " " + "loss:"+ str(loss_avg()))


def train_and_evaluate(model, train_data, val_data, optimizer_1,optimizer_2,loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
#     print(train_data)
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    savecoeff = 0
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))


        train(model, optimizer_1,optimizer_2, loss_fn, train_data,
              metrics, params)
        val_metrics = evaluate(
            model, loss_fn, val_data, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        
        
        
        # Save weights
        savecoeff = savecoeff+1
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optimw_1_dict': optimizer_1.state_dict(),
                               'optimw_2_dict': optimizer_2.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
        if savecoeff%5==0:
            fpath = osp.join(model_dir, 'model'+str(epoch)+'.pth.tar-' + str(epoch))
            torch.save({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_1_dict': optimizer_1.state_dict(),
                               'optim_2_dict': optimizer_2.state_dict()}, fpath)
        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)



if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()


    with open("/home/hesam/owncode/params.json", "r") as read_file:
        data = json.load(read_file)

    params = BertConfig.from_dict(data)
    # print(params)
    # exit(0)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    
    data_path = '/home/hesam/owncode/data/train.jsonl'
    data_path_vald = '/home/hesam/owncode/data/dev.jsonl'
    img_dir   = '/home/hesam/owncode/data/'
    a = HatefulMemesDataset(data_path,img_dir)
    a_vald = HatefulMemesDataset(data_path_vald,img_dir)
    print(a)

    batch_size = 16
    params.batch_size = batch_size 
    # validation_split = .2
    shuffle_dataset = True
    random_seed= 42


#     dataset_size = len(a)
    indices = list(range(len(a)))
#     split = int(500)
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(a, batch_size=batch_size)
#                                                    ,sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(a_vald, batch_size=batch_size)



    # specify the train and val dataset sizes
    params.train_size = len(train_loader)#train_data['size']
    params.val_size = len(validation_loader)#val_data['size']


    logging.info("- done.")



    model = net.VisualBERTForClassification(params).cuda() if params.cuda else net.VisualBERTForClassification(params)



    optimizer_1 =torch.optim.Adam(model.parameters(), lr= 1.0e-03, eps=1.0e-08)
    optimizer_2 =torch.optim.Adam([model.odr], lr= 1.0e-02, eps=1.0e-08)

    loss_fn = net.loss_fn
    metrics = net.metrics


    # Train the model

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_loader, validation_loader, optimizer_1,optimizer_2, loss_fn, metrics, params,            args.model_dir,args.restore_file)
        
        
