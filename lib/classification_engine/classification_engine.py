from ..config import app as app_config
from ..config import classification_cfg as classification_config
from .cls_module.model import DrugClassificationModel_v2
import cv2
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

from .cls_module.utils import *
num_classes = 107+1
use_diagnose = True
use_drugname = True
use_additional = False
backbone = "swin"
BERT_MODEL = "vinai/phobert-base"     

class ClassificationEngine(object):
    def __init__(self, ckpt_dir, device, drugname_2_id_path, doctor_2_id_path):
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.cls_model = DrugClassificationModel_v2(backbone=backbone, num_classes=num_classes, bert_model=BERT_MODEL, 
                                       use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
        load_checkpoint(torch.load(self.ckpt_dir, map_location=self.device), self.cls_model)
        self.cls_model = self.cls_model.to(self.device)
        self.cls_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False, lower_case=True)
        self.drugname_2_id = load_json(drugname_2_id_path)
        self.doctor_2_id = load_json(doctor_2_id_path)
        self.infer = Inference(self.cls_model, tokenizer=self.tokenizer, drugname_2_id=self.drugname_2_id, doctor_2_id=self.doctor_2_id)

    def predict_dataframe(self, dataframe, img_folder, saved_dir):
        _, _, dataframe = preprocessing_dataframe(dataframe)

        is_label = False
        if "id" in dataframe.keys() and dataframe['id'].iloc[0] is int:
            is_label = True

        if is_label:
            predictions = []
            true_targets = []
            result_df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
        else:
            result_df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])

        for idx in tqdm(range(len(dataframe))):
            image_name = dataframe["image_name"].iloc[idx]
            image_path = os.path.join(img_folder, image_name)
            drug_names = dataframe["drugname"].iloc[idx].split("[SEP]")
            doctor = dataframe["doctor"].iloc[idx]
            bbox = dataframe["bbox"].iloc[idx]
            diagnose = dataframe["diagnose"].iloc[idx]
            quantity = dataframe["encode_quantity"].iloc[idx].split()
            prescription_mapping = dataframe["prescription_mapping"].iloc[idx].split()
            
            pred_idx, prob = self.infer._run_classify(image_path=image_path, diagnose=diagnose, 
                                                    drugnames=drug_names, bbox=bbox, doctor=doctor, 
                                                    quantity=quantity, device=self.device)
            if (not is_empty(except_108(prescription_mapping)) and pred_idx not in except_108(prescription_mapping)) or prob<0.3:
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
                label = dataframe["id"].iloc[idx]
                predictions.append(pred_idx)
                true_targets.append(label)
                
                if pred_idx!=label:
                    print("Fail case: {}; pred: {}; gt: {}; prob: {}; mapping: {}".format(image_name, pred_idx, label, prob, except_108(prescription_mapping)))
                result_df.loc[idx] = [image_name, pred_idx, prob, x1, y1, x2, y2, label]
            else:
                image_name = "_".join(image_name.split("_")[:4]) + ".jpg"
                result_df.loc[idx] = [image_name, pred_idx, prob, x1, y1, x2, y2]\

        # saved_dir = "/".join(self.ckpt_dir.split("/")[:-1])
        if is_label:
            report = classification_report(true_targets, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(saved_dir,"report.csv"))
            acc, f1 = cls_metrics(true_targets, predictions)
            print("\t Validation accuracy score:", acc)
            print("\t Validation f1 score:", f1)

        result_df.to_csv(os.path.join(saved_dir,"results.csv"), index=False)
        
        

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
        
        log_prob, idx = torch.max(preds, 1)
        prob = torch.exp(log_prob).item()
        
        return idx.item() ,prob
        
    def _transform_image(self, image_path):
        image = cv2.imread(image_path)
        image = padding_image(image, new_size=(self.height, self.width))
        image = image.transpose([2,0,1]) / 255

        return torch.Tensor(image)

    def _transform_drugname(self, drug_names):
        one_hot_vector = np.zeros(len(self.drugname_2_id))
        for drug in drug_names:
            try:
                one_hot_vector[self.drugname_2_id[drug]] = 1
            except:
                pass
        return torch.Tensor(one_hot_vector)

    def _transform_diagnose(self, diagnose):
        token_input = self.tokenizer.encode(diagnose, add_special_tokens=False)
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