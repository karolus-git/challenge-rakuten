import requests
import json
import numpy as np
from utils import preprocessors
import transformers
import PIL
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def fusion(texts, images, model):

    transform = preprocessors.transform_image()
    images_transformed = torch.stack([transform(image) for image in images])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "camembert-base", 
        use_fast=True)

    texts = [preprocessors.standardization_text(text) for text in texts]
    texts = np.array(texts).astype(str).tolist()

    tokens = tokenizer(
        texts, 
        padding="max_length",
        truncation=True,
        return_tensors="pt"
        )  

    ids = tokens.input_ids.numpy()
    attentions = tokens.attention_mask.numpy().astype(bool)
    ids_bag = torch.tensor(ids[attentions], dtype=torch.int64)
    offsets_bag = (ids_bag == 5).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        outputs = model(ids_bag, offsets_bag, images_transformed)
    outputs_numpy = outputs.detach().numpy()

    return softmax(outputs_numpy)

def image(images, model):

    transform = preprocessors.transform_image()

    images_transformed = torch.stack([transform(image) for image in images])

    with torch.no_grad():
        outputs = model(torch.Tensor(images_transformed))
        outputs_numpy = outputs.detach().numpy()

    return softmax(outputs_numpy)

def text(texts, model):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "camembert-base", 
        use_fast=True)

    texts = [preprocessors.standardization_text(text) for text in texts]
    texts = np.array(texts).astype(str).tolist()

    tokens = tokenizer(
        texts, 
        padding="max_length",
        truncation=True,
        return_tensors="pt"
        )  

    ids = tokens.input_ids.numpy()
    attentions = tokens.attention_mask.numpy().astype(bool)
    ids_bag = torch.tensor(ids[attentions], dtype=torch.int64)
    offsets_bag = (ids_bag == 5).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        outputs = model(ids_bag, offsets_bag)
    outputs_numpy = outputs.detach().numpy()

    return softmax(outputs_numpy)