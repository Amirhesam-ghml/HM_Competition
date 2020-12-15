

import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch                    
import torchvision
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from PIL import Image

class HatefulMemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,

    ):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        self.img_dir = img_dir


    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]

        image = Image.open(
            self.img_dir+self.samples_frame.loc[idx, "img"]
        ).convert("RGB")
        

        image_dim = 224
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(image_dim, image_dim)
                ),        
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        
        image = image_transform(image)
        
        
        text = self.samples_frame.loc[idx, "text"]
        tokenizer = self.tokenizer
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(indexed_tokens)<64:
            indexed_tokens += [0] * (64 - len(indexed_tokens))
        else:
            indexed_tokens = indexed_tokens[0:64]

        
        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [self.samples_frame.loc[idx, "label"]]
            ).long().squeeze()
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text, 
                "label": label,
                "Token_ids": torch.FloatTensor(indexed_tokens)
                
            }
        else:
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text,
                "Token_ids": indexed_tokens
            }
        
        return sample