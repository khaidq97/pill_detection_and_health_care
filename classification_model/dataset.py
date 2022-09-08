import os
from pydoc import doc
from turtle import width
import numpy as np
import torch
import cv2
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torchvision.transforms as transforms 
from PIL import Image

from  utils import *

transform_fn = transforms.Compose([
                           transforms.RandomRotation(degrees=(0, 15)),
                           transforms.RandomHorizontalFlip(p=0.5)])

class VAIPEDatsets():
    def __init__(self, image_folder, dataframe ,tokenizer, num_classes, drugname_2_id, doctor_2_id, height=320, width=320, max_len=11):
        self.image_folder = image_folder
        self.dataframe = dataframe
        self.index = self.dataframe.index.values

        self.tokenizer = tokenizer
        self.height = height
        self.width = width
        self.max_len = max_len
        self.num_classes = num_classes
        self.drugname_2_id = drugname_2_id
        self.doctor_2_id = doctor_2_id

    def __len__(self):
        return len(self.index)

     # encode drug name to vector
    def encode_drug_name(self, drug_names):
        one_hot_vector = np.zeros(len(self.drugname_2_id))
        for drug in drug_names:
            one_hot_vector[self.drugname_2_id[drug]] = 1
        return one_hot_vector
    
    # encode doctor name to vector
    def encode_doctor_name(self, doctor_name):
        try:
            return np.array(one_hot_fn(self.doctor_2_id[doctor_name], num_dim=len(self.doctor_2_id)))
        except:
            return np.zeros(len(self.doctor_2_id))
        
    # processing diagnose to train with BERT
    def encode_diagnose(self, diagnose):
        token_input = self.tokenizer.encode(diagnose, add_special_tokens=True)
        mask = [1] * len(token_input)
        token_type_ids = [0] * len(token_input)

        padding_len_input = 256 - len(token_input)
        input_ids = token_input + ([0] * padding_len_input)
        mask = mask + ([0] * padding_len_input)
        token_type_ids = token_type_ids + ([0] * padding_len_input)

        return np.array((np.array(input_ids), np.array(mask), np.array(token_type_ids)))

    def __getitem__(self, idx):
        image_names = self.dataframe["image_name"].iloc[idx]
        image = cv2.imread(os.path.join(self.image_folder, image_names))
        image = Image.fromarray(image)
        image = transform_fn(image)
        image = np.array(image)
        image = padding_image(image, new_size=(self.height, self.width))
        # cv2.imwrite(os.path.join("check_data", image_names),image)
        image = image.transpose([2,0,1]) / 255
        
        drug_names = self.dataframe["drugname"].iloc[idx].split("[SEP]")
        doctor = self.dataframe["doctor"].iloc[idx]
        bbox = self.dataframe["bbox"].iloc[idx]
        diagnose = self.dataframe["diagnose"].iloc[idx]
        quantity = self.dataframe["encode_quantity"].iloc[idx].split()
        label = self.dataframe["id"].iloc[idx]

        bbox = bbox.split(" ")
        x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[0]))+int(float(bbox[2])), int(float(bbox[1]))+int(float(bbox[3]))

        # encode diagnose
        diagnose_input = self.encode_diagnose(diagnose)
        # encode drug_names
        drug_names_input = self.encode_drug_name(drug_names)
        # encode doctor
        doctor_input = self.encode_doctor_name(doctor)
        # doctor_input = doctor
        # quantity
        quantity_input = np.array([float(i) for i in quantity])
        
        bbox_input = np.array(((x2-x1) / self.width, (y2-y1)/ self.height)) 
        targets = np.array(one_hot_fn(label, num_dim=self.num_classes))

        return {
            "image_input": torch.Tensor(image),
            "diagnose_input": torch.as_tensor(diagnose_input),
            "bbox_input": torch.Tensor(bbox_input),
            "drugnames_input": torch.Tensor(drug_names_input),
            "doctor_input": torch.Tensor(doctor_input),
            "quantity_input": torch.Tensor(quantity_input),
            "targets": torch.Tensor(targets)
        }
        
if __name__ == "__main__":
    dataframe = pd.read_csv("pill_csv_fix_bbox/public_train_encoded+gen_case.csv")
    train_df, val_df, dataframe = preprocessing_dataframe(dataframe)
    
    print(len(dataframe['doctor'].unique()))
    
    num_classes = 107 + 1
    BERT_MODEL = "vinai/phobert-base"
    # xlm-roberta-base
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False, lower_case=True)
    
    drugname_2_id = load_json(os.path.join("mapping","drugname_2_id.json"))
    doctor_2_id = load_json(os.path.join("mapping", "doctor_2_id.json"))
    
    print("drugname_2_id", len(drugname_2_id))
    print("doctor_2_id", len(doctor_2_id))
    
    data_path = "/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection/public_train"
    train_folder = os.path.join(data_path, "train+gen_case+val_v5")
    train_dataset = VAIPEDatsets(train_folder, train_df, tokenizer, num_classes, drugname_2_id, doctor_2_id)
    
    list(train_dataset)
    # a = next(iter(train_dataset))
    # print("image_input:",a["image_input"].shape)
    # print("diagnose_input:",a["diagnose_input"].shape)
    # print("bboxes_input:",a["bbox_input"].shape)
    # print("drugnames_input:",a["drugnames_input"].shape)
    # print("doctor_input:",a["doctor_input"].shape)
    # print("quantity_input:",a["quantity_input"].shape)
    # print("targets:",a["targets"].shape)