# Copyright (c) Facebook, Inc. and its affiliates.
# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy

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


class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        # embedding_strategy="plain",
        # bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim

        self.bert_out = BertModel.from_pretrained("bert-base-uncased")
        self.embeddings = myVisio(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
    ):


        Bert_embeddings = self.bert_out(input_ids,token_type_ids,attention_mask[:,0:128])


        v_embeddings = self.embeddings(
            # input_ids,
            # token_type_ids,
            visual_embeddings=visual_embeddings,
            position_embeddings_visual=position_embeddings_visual,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        #Concate the two:
        embeddings = torch.cat(
            (Bert_embeddings[0], v_embeddings), dim=1
        )  # concat the visual embeddings after the attentions


        # if self.output_attentions:
        encoded_layers = self.encoder(embedding_output)

        sequence_output = encoded_layers[0]
        attn_data_list = encoded_layers[1:]
        pooled_output = self.pooler(sequence_output)
        return encoded_layers, pooled_output, attn_data_list





class VisualBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        self.bert = VisualBERTBase(
            self.config,
            visual_embedding_dim=self.config.visual_embedding_dim,

        )

        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # if self.config.training_head_type == "nlvr2":
        #     self.bert.config.hidden_size *= 2

        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
        )

        self.init_weights()

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

        sample_list.visual_embeddings = image_feat_variable
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
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        
        sequence_output, pooled_output, attention_weights = self.bert(
            sample_list.input_ids,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,
        )


        output_dict = {}
        # if self.output_attentions:
        output_dict["attention_weights"] = attention_weights

        # if self.output_hidden_states:
        output_dict["sequence_output"] = sequence_output
        output_dict["pooled_output"] = pooled_output

        # if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
        index_to_gather = input_mask.sum(1) - 2
        pooled_output = torch.gather(
            sequence_output,
            1,
            index_to_gather.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits
        return output_dict



def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, labels)




def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
