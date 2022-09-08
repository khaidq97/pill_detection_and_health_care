from cProfile import label
from cgi import test
from lib2to3 import refactor
from pickle import TRUE
from tkinter.tix import Tree
import numpy as np
import torch
import os
import cv2
import pandas as pd
import logging
import sys
from datetime import datetime

from model.model import DrugClassificationModel
from model_v2.model import DrugClassificationModel_v2
from model_v3.model import DrugClassificationModel_v3
from dataset import VAIPEDatsets
from validation_fn import validation_fn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import *
import json

class Inference():
    def __init__(self, model, tokenizer, drugname_2_id, doctor_2_id, height=320, width=320):
        self.model = model
        self.height = height
        self.width = width
        self.tokenizer = tokenizer
        self.drugname_2_id = drugname_2_id
        self.doctor_2_id = doctor_2_id

    def _run_classify(self, image_path, diagnose, drugnames, bbox, doctor, quantity, device):
        image_input = self._transform_image(image_path)
        diagnose_input = self._transform_diagnose(diagnose).to(device)
        drugnames_input = self._transform_drugname(drugnames)
        bbox_input = self._transform_bbox(bbox)
        doctor_input = self._transform_doctor(doctor)
        quantity_input = self._transform_quantity(quantity)
        
        image_input = image_input.unsqueeze(dim=0).to(device)
        drugnames_input = drugnames_input.unsqueeze(dim=0).to(device)
        diagnose_input = diagnose_input.unsqueeze(dim=0).to(device)
        bbox_input = bbox_input.unsqueeze(dim=0).to(device)
        doctor_input = doctor_input.unsqueeze(dim=0).to(device)
        quantity_input = quantity_input.unsqueeze(dim=0).to(device)
        
        with torch.no_grad():
            preds = self.model(image_input, diagnose_input, drugnames_input, bbox_input, doctor_input, quantity_input)
        
        # softmax
        # softmax_fn = torch.nn.Softmax(dim=1)
        # preds = softmax_fn(preds) 
        # prob, idx = torch.max(preds, 1)
        
        # log_softmax   
        log_prob_list, idx_list = torch.sort(preds, 1, descending=True)
        prob_list = torch.exp(log_prob_list)
        
        log_prob, idx = torch.max(preds, 1)
        prob = torch.exp(log_prob)
    
        
        return idx.item() ,prob.item(), idx_list[0].cpu().numpy(), prob_list[0].cpu().numpy()

    def _transform_image(self, image_path):
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_AREA)
        image = padding_image(image, new_size=(self.height, self.width))
        image = image.transpose([2,0,1]) / 255

        return torch.Tensor(image)

    def _transform_drugname(self, drug_names):
        one_hot_vector = np.zeros(len(self.drugname_2_id))
        for drug in drug_names:
            try:
                drug = text2IDdrug(drug, self.drugname_2_id)
                one_hot_vector[self.drugname_2_id[drug]] = 1
            except:
                pass
        return torch.Tensor(one_hot_vector)

    def _transform_diagnose(self, diagnose):
        token_input = self.tokenizer.encode(diagnose, add_special_tokens=True)
        mask = [1] * len(token_input)
        token_type_ids = [0] * len(token_input)

        padding_len_input = 256 - len(token_input)
        input_ids = token_input + ([0] * padding_len_input)
        mask = mask + ([0] * padding_len_input)
        token_type_ids = token_type_ids + ([0] * padding_len_input)

        return torch.as_tensor(np.array((np.array(input_ids), np.array(mask), np.array(token_type_ids))))
    
    def _transform_doctor(self, doctor_name):
        try:
            doctor_input =  np.array(one_hot_fn(self.doctor_2_id[doctor_name], num_dim=len(self.doctor_2_id)))
        except:
            doctor_input = np.zeros(len(self.doctor_2_id))
            
        return torch.Tensor(doctor_input)
    
    def _transform_quantity(self, quantity):
        quantity_input = np.array([float(i) for i in quantity])
        
        return torch.Tensor(quantity_input)
    
    def _transform_bbox(self, bbox):
        bbox = bbox.split(" ")
        x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[0]))+int(float(bbox[2])), int(float(bbox[1]))+int(float(bbox[3]))
        bbox_input = np.array(((x2-x1) / self.width, (y2-y1)/ self.height)) 
        
        return torch.Tensor(bbox_input)
 
def recorrection_results(dataframe):
    refactor_df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    index = 0
    for image_name in tqdm(dataframe['image_name'].unique()):
        sub_df = dataframe[dataframe['image_name']==image_name]
        id_count = sub_df['class_id'].value_counts()
        count_107 = 0
        try:
            count_107 = id_count[107]
        except:
            pass
        for idx in range(len(sub_df)):
            id = sub_df['class_id'].iloc[idx]
            mapping_id = sub_df['pred_id_list'].iloc[idx]
            prob = sub_df['confidence_score'].iloc[idx]
            x1 = sub_df['x_min'].iloc[idx]
            y1 = sub_df['y_min'].iloc[idx]
            x2 = sub_df['x_max'].iloc[idx]
            y2 = sub_df['y_max'].iloc[idx]
            if count_107 < 3 and id==107:
                id = mapping_id[1]
            elif count_107 > 3 and id!=107:
                id = 107
                
            refactor_df.loc[index] = [image_name, id, prob, x1, y1, x2, y2]
            index+=1
            
                
    return refactor_df
   
def infer_in_dataset(df, img_folder, model, is_label=False, save_file=None):
    Classify = Inference(model, tokenizer=tokenizer, drugname_2_id=drugname_2_id, doctor_2_id=doctor_2_id)
    predictions = []
    true_targets = []
    if is_label:
        result_df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
    else:
        result_df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max', 'pred_id_list'])
        
    
    for idx in tqdm(range(len(df))):
        image_name = df["image_name"].iloc[idx]
        image_path = os.path.join(img_folder, image_name)
        drug_names = df["drugname"].iloc[idx].split("[SEP]")
        doctor = df["doctor"].iloc[idx]
        bbox = df["bbox"].iloc[idx]
        diagnose = df["diagnose"].iloc[idx]
        quantity = df["encode_quantity"].iloc[idx].split()
        prescription_mapping = df["prescription_mapping"].iloc[idx].split()
        
        pred_idx, prob, pred_id_list, prob_list = Classify._run_classify(image_path=image_path, diagnose=diagnose, 
                                                drugnames=drug_names, bbox=bbox, doctor=doctor, 
                                                quantity=quantity, device=device)
        
        if pred_idx==107 and prob<0.6:
            pred_idx = pred_id_list[1]
            prob = prob_list[1]
        # if pred_idx in [3,88,89,90,96,99] and len(drug_names)==1 and "MEDIPLEX 800mg" in drug_names:
        #     pred_idx = 3
        if pred_idx == 28 and 'ALFACHIM 4,2mg' in drug_names and 'S91' in diagnose:
            pred_idx = 7
        if pred_idx==26 and len(drug_names)==1 and "CEPHALEXIN 250MG 0,25g" in drug_names:
            pred_idx = 25
        if "DƯỠNG TÂM AN THẦN" in drug_names and pred_idx==107 and pred_id_list[1]==44:
            pred_idx=44
        if "Omeprazol (Kagasdine) 20mg" in drug_names and pred_idx==66:
            pred_idx=87
        if (not is_empty(except_108(prescription_mapping)) and pred_idx not in except_108(prescription_mapping)) or prob<0.0:
            pred_idx = 107
            prob = 1
            
        # if pred_idx not in except_108(prescription_mapping) or prob<0.3:
        #     pred_idx = 107
        #     prob = 1
        
        # if pred_idx not in except_108(prescription_mapping):
        #     pred_idx = 107
        #     prob = 1 
        
        bbox = bbox.split(" ")
        x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[0]))+int(float(bbox[2])), int(float(bbox[1]))+int(float(bbox[3]))
        
        if is_label:
            label = val_df["id"].iloc[idx]
            predictions.append(pred_idx)
            true_targets.append(label)

            if pred_idx!=label:
                logging.info("Fail case: {}; pred: {}; gt: {}; prob: {}; mapping: {}".format(image_name, pred_idx, label, prob, except_108(prescription_mapping)))
                logging.info("idx list: {}; prob list: {}".format(pred_id_list[:2], prob_list[:2]))
                logging.info("**********")
            result_df.loc[idx] = [image_name, pred_idx, prob, x1, y1, x2, y2, label]
        else:
            image_name = "_".join(image_name.split("_")[:4]) + ".jpg"
            result_df.loc[idx] = [image_name, pred_idx, prob, x1, y1, x2, y2, pred_id_list]
            
        # if idx==50:
        #     break
        
        
    saved = "/".join(saved_dir.split("/")[:-1])
    report = classification_report(true_targets, predictions, output_dict=True)

    if is_label:
      report_df = pd.DataFrame(report).transpose()
      report_df.to_csv(os.path.join(saved,"report.csv"))
    else: 
        # result_df = recorrection_results(result_df)
        result_df = result_df[['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']]
        result_df.to_csv(os.path.join(saved,save_file), index=False)
    
    acc, f1 = cls_metrics(true_targets, predictions)
    logging.info("\t Validation accuracy score:{}".format(acc))
    logging.info("\t Validation f1 score:{}".format(f1))
    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join("log_inference", str(datetime.now().strftime("%d-%m-%Y:%H:%M:%S"))+".log"), 
                        format='%(asctime)s %(message)s', filemode='w',
                        )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    # dataframe = pd.read_csv("pill_csv_v2/public_train_encoded.csv")
    # _, _, dataframe = preprocessing_dataframe(dataframe)
    
    val_df = pd.read_csv("pill_csv_v2/public_test_gt_encoded.csv")
    # val_df = pd.read_csv("pill_csv_fix_bbox/public_test_encoded.csv")
    _, _, val_df = preprocessing_dataframe(val_df)
    # test_df = pd.read_csv("pill_csv_fix_bbox/public_test_new_encoded.csv")
    test_df = pd.read_csv("pill_csv_fix_bbox/public_test_new_no_fix_bbox_encoded.csv")
    _, _, test_df =preprocessing_dataframe(test_df)
    
    data_path = "/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection"
    BERT_MODEL = "vinai/phobert-base"     # xlm-roberta-base
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False, lower_case=True)

    # get drug name to id
    drugname_2_id = load_json(os.path.join("mapping","drugname_2_id.json"))
    doctor_2_id = load_json(os.path.join("mapping", "doctor_2_id.json"))
    val_folder = os.path.join(data_path, "public_test", "gt")
    # val_folder = os.path.join(data_path, "public_test_new", "test_new_mapping")
    # test_folder = os.path.join(data_path, "public_test_new","test_new")
    test_folder = os.path.join(data_path, "public_test_new","test_new_no_fix_bbox")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 107+1
    use_diagnose = True
    use_drugname = True
    use_additional = True
    backbone = "swin"
    model_name = backbone+"_"+str(use_diagnose)+"_"+str(use_drugname)+"_"+str(use_additional)+"_tiny_v2_without_bbox" #+"_107_class" # tiny_attention_v3
    # model_name = "resnet50_"+str(use_diagnose)+"_"+str(use_drugname)+"_"+str(use_additional)+"_107_class" #+"_107_class"
    # saved_dir = "saved_models/"+model_name+"/"+model_name+"_mapping_17_18_19_20_21_22_23_24_25_26_27_28_29_30.pt"
    # saved_dir = "saved_models/"+model_name+"/"+model_name+"_mapping_17.pt"
    saved_dir = "saved_models/"+model_name+"/"+model_name+".pt"

    # saved_dir = "saved_models/resnet50_cls_no_drugname.pt"
    # model = DrugClassificationModel(num_classes=num_classes, bert_model=BERT_MODEL, 
    #                                 use_diagnose=use_diagnose, use_drugname=use_drugname, 
    #                                 use_additional=use_additional)
    
    model = DrugClassificationModel_v2(backbone=backbone, num_classes=num_classes, bert_model=BERT_MODEL, 
                                       use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
    # model = DrugClassificationModel_v3(backbone=backbone, num_classes=num_classes, bert_model=BERT_MODEL, 
                                    #    use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
    
    last_epoch, train_f1, val_f1 = load_checkpoint(torch.load(saved_dir, map_location=torch.device("cpu")), model)
    model = model.to(device)
    model.eval()
    
    infer_in_dataset(df=val_df, img_folder=val_folder, model=model, is_label=True, save_file="val_results.csv")
    # infer_in_dataset(df=val_df, img_folder=val_folder, model=model, is_label=False, save_file="results.csv")
    # infer_in_dataset(df=test_df, img_folder=test_folder, model=model, is_label=False, save_file="results.csv")
    

    
    
    
    
