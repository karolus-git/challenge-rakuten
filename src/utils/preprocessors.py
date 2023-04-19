import torch
import torchtext
import re
import unidecode
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
import logging
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TODO Move this as image model variables
T_MEAN = [0.485, 0.456, 0.406]
T_STD = [0.229, 0.224, 0.225]

def get_tokenizer():
    """Get a tokenizer 

    Returns:
        _type_: tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        'camembert-base',
        use_fast=True
        )

    logger.info(f"tokenizer camembert initialized")
    
    return tokenizer

def tokenize_batch(samples, **kwargs):

    tokenizer = kwargs.get("tokenizer")

    texts = [sample["text"] for sample in samples]
    images = [sample["image"].clone().detach() for sample in samples]
    labels = torch.tensor([sample["label"] for sample in samples])

    texts = [standardization_text(text) for text in texts]

    tokens = tokenizer(texts, 
        padding="max_length",
        truncation=True,
        return_tensors="pt")

    ids = tokens.input_ids.numpy()
    attentions = tokens.attention_mask.numpy().astype(bool)
    ids_bag = torch.tensor(ids[attentions], dtype=torch.int64)
    offsets_bag = (ids_bag == 5).nonzero(as_tuple=True)[0]

    images = torch.stack(images)

    return {
        "input_ids": tokens.input_ids, 
        "attention_mask": tokens.attention_mask, 
        "labels": labels, 
        "images": images, 
        "texts": texts,
        "input_ids_bag": ids_bag, 
        "offsets_bag" : offsets_bag
        }


def transform_image(crop_shape=400, resize_shape=224, with_crop=True):
    """Image transformer. The image is cropped if asked, and then resized, converted to a tensor and normalized.

    Args:
        crop_shape (int, optional): size after cropping. Defaults to 400.
        resize_shape (int, optional): sife after resizing. Defaults to 224.
        with_crop (bool, optional): should cropping be applied? Defaults to True.

    Returns:
        T.Compose: bag of actions to transform an image
    """
    actions = []
    actions.append(T.Resize((resize_shape,resize_shape)))           
    actions.append(T.ToTensor())
    actions.append(T.Normalize(mean=T_MEAN, std=T_STD))

    # if with_crop:
    #     actions.append(T.CenterCrop(crop_shape))

    logger.info("actions for image transformation setted")

    composition = T.Compose(actions)

    return composition
    
def invert_transform_image(resize_shape=500):
    """Image invert transformer. 

    Args:
        crop_shape (int, optional): size after cropping. Defaults to 400.
        resize_shape (int, optional): sife after resizing. Defaults to 224.
        with_crop (bool, optional): should cropping be applied? Defaults to True.

    Returns:
        T.Compose: bag of actions to transform an image
    """
    actions = []

    actions.append(T.Normalize(
        mean=(-1 * np.array(T_MEAN) / np.array(T_STD)), 
        std=1/np.array(T_STD))
    )
    actions.append(T.Lambda(lambda x: (x*255).int()))

    logger.info("actions for image inverse transformation setted")
    return T.Compose(actions)


def standardization_text(x):
    """Standardize text.

    - Remove htlm
    - Remove &#39;
    - Remove mutli-spaces
    - Lower

    Args:
        x (str): text to standardize

    Returns:
        str: standardized text
    """

    x_raw = x

    #Remove html tags
    x = re.sub("<[^>]+>", " ", x)

    #Remove `
    x = re.sub('&#39;', " ", x)
    
    #Remove '
    x = re.sub("'", " ", x)
    
    #Remove '
    x = re.sub("[ ]+", " ", x)

    #Apply lowering
    x = x.lower()

    #Decode
    x = unidecode.unidecode(x)

    logger.debug(f"text standardized from <TEXT>{x_raw} to <STANDARDIZE>{x}")

    return x

