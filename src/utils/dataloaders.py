import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import os
#from .preprocessors import preprocessing_fn_texts, preprocessing_fn_tfidf, texts_load_and_adapt_vectorization_layer, preprocessing_fn_images, texts_standardization
import torch
import torchvision
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import functools
import PIL
import logging

from . import preprocessors


logger = logging.getLogger(__name__)


class ChallengeDataModule(pl.LightningDataModule):

    def __init__(self,
        root_path: str="",
        data_file: str="",
        label_file: str="",
        transform_image: bool=True,
        batch_size : int=32,
        num_workers : int=1,
        shuffle : bool=False,
        samples : int=0,
        splits={},
        crop_shape:int=500,
        resize_shape:int=500,
        ):
        
        super().__init__()

        #Model will be saved in checkpoint
        self.save_hyperparameters(ignore=['model'])

        self.root_path = root_path
        self.data_file = data_file
        self.label_file = label_file
        self.transform_image = transform_image
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.samples = samples
        self.splits = splits
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape

        assert np.array(splits).sum().round(3) == 1.0, "The sum of splits should be 1"

    def prepare_data(self):

        #Get the data
        self.dataset = self.read_data()

        #Get the vocab and the associated tokenizer
        self.tokenizer = preprocessors.get_tokenizer()

        logger.info("data preparation done")
        return self

    def setup(self, stage: str):

        #Create splits
        datasplits = torch.utils.data.random_split(
            self.dataset, self.splits
        )

        #Affect split to different datasets
        self.train_dataset = datasplits[0]
        self.val_dataset = datasplits[1]
        self.test_dataset = datasplits[2]
        

    # def get_vocab(self):
    #     return self.vocab

    def read_data(self):
        
        dataset = DatasetTorch(
            root_path=self.root_path,
            data_file=self.data_file,
            label_file=self.label_file,
            crop_shape=self.crop_shape,
            resize_shape=self.resize_shape,
            samples=self.samples
        )
        logger.info("data converted from torch to lightning")
        return dataset

    def train_dataloader(self,):
        logger.info("train : conversion from dataset to dataloader")

        return DataLoader(self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=self.shuffle, 
            collate_fn=functools.partial(
                preprocessors.tokenize_batch, 
                    tokenizer=self.tokenizer)
            )
    
    def test_dataloader(self,):
        logger.info("test : conversion from dataset to dataloader")
        return DataLoader(self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=self.shuffle, 
            collate_fn=functools.partial(
                preprocessors.generate_batch, 
                    tokenizer=self.tokenizer)
            )
    
    def val_dataloader(self,):
        logger.info("val : conversion from dataset to dataloader")

        return DataLoader(self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=self.shuffle, 
            collate_fn=functools.partial(
                preprocessors.tokenize_batch, 
                    tokenizer=self.tokenizer)
            )

class DatasetTorch(torch.utils.data.Dataset):
    def __init__(self,
        root_path: str="",
        data_file: str="",
        label_file: str="",
        transform_image: bool=True,
        crop_shape:int=500,
        resize_shape:int=500,
        samples:int=0,
    ):

        #Attributes
        self.root_path = root_path
        self.transform_image = transform_image
        self.transform_image_process = preprocessors.transform_image(
            crop_shape=crop_shape,
            resize_shape=resize_shape,
            with_crop=True
            )

        #Make links to csv 
        features_path = Path(root_path, data_file)
        labels_path = Path(root_path, label_file)

        #Load csv files
        df = pd.read_csv(features_path, index_col="index").astype(str).fillna(" ")
        if samples:
            df = df.head(samples)

        #Build links to images
        df["links"] = (root_path + "/images/image_train/image_" + df.imageid +"_product_" + df.productid + ".jpg").values

        #Merge columns
        df["text"] = df.designation +  " " + df.description
        
        #Add two columns for words counts
        df["words_designation"] = df.designation.apply(lambda x : len(x))
        df["words_description"] = df.description.apply(lambda x : len(x))

        #Add the labels
        labels = pd.read_csv(labels_path, index_col="Unnamed: 0")
        if samples:
            labels = labels.head(samples)
        df["labels"] = labels.prdtypecode

        self.le = LabelEncoder()
        df["labels"] = self.le.fit_transform(df["labels"])

        torch.save(self.le, 'obj_labelencoder.pth')

        #Remove useless columns
        self.df = df.drop(["description", "designation", "imageid", "productid"], axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        logger.debug(f"get items from datasettorch: {idx}")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Get the features for this idx
        label = self.df.iloc[idx, -1]
        img_link = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]

        #Handle image
        image = PIL.Image.open(img_link)
        if self.transform_image:
            image = self.transform_image_process(image)

        return {
            "text" : text,
            "image" : image,
            "label" : label
        }