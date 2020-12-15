# Copyright (c) Facebook, Inc. and its affiliates.
# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader

import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (

    BertConfig,
    BertEncoder,
    BertForPreTraining,
    BertLayer,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
import numpy as np
import torchvision
from mmf.common.registry import registry
from mmf.models import BaseModel
# from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from Module.Visio_embeddings import myVisio
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)

from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import json






class VisualBERTForClassification(nn.Module):
    def __init__(self, config,visual_embedding_dim=512):
        super().__init__()
        visual_embedding_dim=512
        self.config = config
#         print("here")
        self.bert_out = BertModel.from_pretrained("bert-base-uncased")
        
        model_3 = torchvision.models.resnet152(pretrained=True)
        modules = list(model_3.children())[:-2]
        self.resnet= torch.nn.Sequential(*modules) 
        self.img_seq_1 = nn.Sequential(nn.Linear(49,20),
                                     nn.ReLU(),)
        self.img_seq_2 = nn.Sequential(nn.Linear(2048,1024),
                                       nn.ReLU(),
                                       nn.Linear(1024,768),
                                       nn.ReLU(),)

#         self.embeddings = myVisio(config)
        config.visual_embedding_dim = visual_embedding_dim
        
        
        
        self.wc1 = torch.nn.Parameter(torch.zeros(config.hidden_size,100))
        self.wc2 = torch.nn.Parameter(torch.zeros(100,10))
        self.wc3 = torch.nn.Parameter(torch.zeros(840,256))
        self.odr = torch.nn.Parameter(torch.zeros(1,840)) 
        self.wc4 = torch.nn.Parameter(torch.zeros(256,128))
        
        nn.init.xavier_uniform_(self.wc1, gain=1.0)
        nn.init.xavier_uniform_(self.wc2, gain=1.0)
        nn.init.xavier_uniform_(self.wc3, gain=1.0)
#         nn.init.xavier_uniform_(self.odr, gain=1.0)
#         nn.init.normal_(self.odr, mean=0.0, std=1.0)
        nn.init.xavier_uniform_(self.wc4, gain=1.0)
        
        
        self.fusion = torch.nn.Linear(in_features=128,out_features=2)
        
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  
        


    def init_weights(self):
        if self.config.random_initialize is False:
            # if self.bert_model_name is None:
                # No pretrained model, init weights
                # self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)


    def flatten(self, sample_list, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])

        if sample_list.visual_embeddings_type is None:
            if sample_list.image_mask is not None:
                sample_list.visual_embeddings_type = torch.zeros_like(
                    sample_list.image_mask, dtype=torch.long
                )

        if sample_list.image_mask is not None:
            attention_mask = torch.cat(
                (sample_list.input_mask, sample_list.image_mask), dim=-1
            )
            if sample_list.masked_lm_labels is not None:
                assert sample_list.masked_lm_labels.size(
                    -1
                ) == sample_list.input_mask.size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = sample_list.masked_lm_labels.size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[
                    : size_masked_lm_labels[0], : size_masked_lm_labels[1]
                ] = sample_list.masked_lm_labels
                sample_list.masked_lm_labels = new_lm_labels
        else:
            attention_mask = sample_list.input_mask

        sample_list.attention_mask = attention_mask

        return sample_list


    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            "input_ids",
            "token_type_ids",
            "input_mask",
            "image_mask",
            "masked_lm_labels",
            "position_embeddings_visual",
            "visual_embeddings_type",
        ]
        to_be_flattened_dim = ["image_text_alignment", "visual_embeddings"]

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = getattr(image_info, "max_features", None)
        image_feat_variable = getattr(sample_list, "image_feature_0", None)
        image_info = getattr(sample_list, "image_feature_0", None)

        #sample_list.visual_embeddings = image_feat_variable
        sample_list.visual_embeddings = torch.from_numpy(np.squeeze(np.expand_dims(sample_list.image_info_0.cls_prob,0))).cuda()
        #print(sample_list.visual_embeddings.size())
        #exit(0)
        #sample_list.visual_embeddings = torch.randn(8,100,1601).cuda()

        
        sample_list.image_dim = image_dim_variable.cuda()
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2))
                .expand(*visual_embeddings.size()[:-1])
                .cuda()
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        return sample_list


    def forward(
        self,sample_list,
    ):

        
        image_feats = self.resnet(sample_list['image'])
        image_feats = image_feats.view((-1,2048,49))
        image_redim = self.img_seq_1(image_feats)
        image_redim = image_redim.view((-1,20,2048))

        image_redim = self.img_seq_2(image_redim)

        with torch.no_grad():
            self.bert_out.eval()

            Bert_embeddings = self.bert_out(sample_list['Token_ids'].long())
            

        
        Concat  = torch.cat((Bert_embeddings[0], image_redim), dim=1) 
        
        result1 = torch.mm(Concat.view(-1, 768), self.wc1)
        result1 = result1.view(-1, 64+20, 100)
        result1 = nn.ReLU()(result1)
        
        

        
        result2 = torch.mm(result1.view(-1, 100), self.wc2)
        result2 = result2.view(-1, 64+20, 10)
        result2 = nn.ReLU()(result2)
        result2 = result2.view(-1, 840)

        
        result2 = result2 + self.odr #+ (10^-3)*torch.sigmoid(result2)
        result3 = torch.mm(result2, self.wc3)
        result3 = nn.ReLU()(result3)
        
        result4 = torch.mm(result3, self.wc4)
        result4 = nn.ReLU()(result4)
        
        
        result5 = self.fusion(result4)

        output = result5
        

        return output




def loss_fn(outputs, labels,wieghts):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len w
        
        each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
#     print("hello")
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)


    loss2 = 0
    for i in wieghts:
        loss2 = loss2 + torch.norm(i)

    return loss #+ (1/20)*loss2#+ (30000/3001)*(1/1601)*sum(torch.norm(weight,p=2, dim=1)) 
                                                              #(300/1601)*sum(torch.norm(weight,p=2, dim=1) )



def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """


    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(outputs.shape[0])


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

