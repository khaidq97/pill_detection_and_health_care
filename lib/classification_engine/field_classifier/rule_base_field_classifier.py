import os
import glob
import re
import json

from tqdm import tqdm
import cv2
from Levenshtein import ratio as lev_ratio

from ...utils.utils import remove_accents, find_in_line_boxes

class RuleBaseFieldClassifier():
    def __init__(self, mapping_drug_name_path):
        # all class: other, diagnose, drugname, usage, quantity, date
        self.usage_pattern = re.compile(r'(sang|chieu|toi|trua)\s*[0-9]+.*(vien)*$')
        self.usage_2_pattern = re.compile(r'uong:*\s*.*[0-9]+.*')
        self.quantity_pattern = re.compile(r'sl:*.*\s*[0-9]+\s*(vien)$')
        self.date_pattern = re.compile(r'ngay\s*(0*[1-9]|1[0-9]| 2[0-9]|3[0-1])\s*(/|thang)\s*(0*[1-9]|1[0-2])\s*(/|nam)\s*20[0-9][0-9]$')
        # self.drug_name_pattern = re.compile(r'[0-9]+\s*\)\s*[A-Z]+.*([0-9]+(mg|g|mcg))$')
        # self.drug_name_pattern = re.compile(r'[0-9]+\s*\)\s*[A-Z]+.*([0-9]+\s*(mg|g|mcg|ui)).*')
        self.drug_name_pattern = re.compile(r'[0-9]+\s*\)\s.+')
        self.drug_name_2_pattern = re.compile(r'[0-9]+\s*\)*\s+[a-zA-Z]+.*([0-9]+\s*(mg|g|mcg|ui))*.*')
        self.diagnose_pattern = re.compile(r'chan\s*doan.*')
        self.ommited_pattern = re.compile(r'thuoc\s*dieu\s*tri.*')
        self.thresh_y_diff_1 = 10
        self.thresh_y_diff_2 = 10
        self.thresh_text_similarity = 0.8
        self.mapping_drug_name_path = mapping_drug_name_path
    
    def classify(self, text_list, text_box_list, have_returned_json=False):
        label_list = []
        first_diagnose_idx = None
        first_drugname_idx = None
        for i, text in enumerate(text_list):
            text_box = text_box_list[i]
            transformed_text = remove_accents(text)
            transformed_text = transformed_text.strip()
            # if re.match(self.drug_name_pattern, transformed_text) or \
            #     re.match(self.drug_name_2_pattern, transformed_text):
            #     label_list.append('drugname')
            #     if first_drugname_idx is None:
            #         first_drugname_idx = i
            if self.is_drug_name(text):
                label_list.append('drugname')
                if first_drugname_idx is None:
                    first_drugname_idx = i
            elif re.search(self.usage_pattern, transformed_text.lower()) or \
                    re.match(self.usage_2_pattern, transformed_text.lower()):
                label_list.append('usage')
            elif re.match(self.quantity_pattern, transformed_text.lower()):
                label_list.append('quantity')
            elif re.match(self.date_pattern, transformed_text.lower()):
                label_list.append('date')
            elif re.search(self.diagnose_pattern, transformed_text.lower()):
                label_list.append('diagnose')
                first_diagnose_idx = i
            else:
                label_list.append('other')
        
        # # recheck drugname base bbox of quantity text
        # for label, box, text in zip(label_list, text_box_list, text_list):
        #     if label == 'quantity':
        #         # find box is aligned with quantity box
        #         in_line_boxes, in_line_indexes = find_in_line_boxes(box, text_box_list, 
        #                                                             self.thresh_y_diff_1, self.thresh_y_diff_2)
        #         image = cv2.rectangle(image, box[:2], box[2:], color=(0,0,255), thickness=2)
        #         for idx in in_line_indexes:
        #             # print("debug in line: ", text_list[idx])
        #             # print(in_line_boxes[idx])
        #             xmin, ymin, xmax, ymax = text_box_list[idx]
        #             image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
        #             if label_list[idx] != 'drugname':
        #                 label_list[idx] = 'drugname'
        #                 if first_drugname_idx is None:
        #                     first_drugname_idx = idx
        
        # find other diagnose texts: between first diagnose and first drugname
        if first_diagnose_idx and first_drugname_idx:
            if first_drugname_idx - first_diagnose_idx > 1:
                for j in range(first_diagnose_idx, first_drugname_idx):
                    if label_list[j] == 'other':
                        text = text_list[j]
                        transformed_text = remove_accents(text)
                        if not re.match(self.ommited_pattern, transformed_text.lower().strip()):
                            label_list[j] = 'diagnose'
                
        if have_returned_json:
            json_obj = self.convert_json_format(text_list, text_box_list, label_list)
            return json_obj
        return text_list, text_box_list, label_list
    
    def export_json(self, json_name, save_root, text_list, text_box_list_label_list):
        pass
    
    def convert_json_format(self, text_list, text_box_list, label_list):
        json_obj = []
        i = 1
        for text, box, label in zip(text_list, text_box_list, label_list):
            dict_obj = {"id": i, "text": text, "label": label, "box": box}
            json_obj.append(dict_obj)
            i += 1
        
        return json_obj
            
    def test(self, json_root):
        json_paths = glob.glob(os.path.join(json_root, "*.json"))
        total_correct = 0
        fail = 0
        for json_path in tqdm(json_paths):
            image_name = json_path.split("/")[-1].split(".")[0]
            text_list = []
            text_box_list = []
            label_list = []
            with open(json_path) as f:
                json_obj = json.load(f)
            correct = True
            for element in json_obj:
                text = element['text']
                box = element['box']
                label = element['label']
                text_list.append(text)
                text_box_list.append(box)
                label_list.append(label)
            _, _, model_label_list = self.classify(text_list, text_box_list, image_name)
            for j, label in enumerate(label_list):
                if model_label_list[j] != label_list[j]:
                    print("="*100)
                    print(json_path)
                    print("="*100)
                    correct = False
                    print("DEBUG: label: {} | model: {} | text:{}".format(label, model_label_list[j], text_list[j]))

            if correct: total_correct += 1
        acc = total_correct/len(json_path)
        return acc
    
    def is_drug_name(self, text):
        with open(self.mapping_drug_name_path) as f:
            mapping_dict = json.load(f)
        drugname_list = mapping_dict.keys()
        text = text.strip()
        best_similarity = 0
        best_similarity_name = None
        for drug_name in drugname_list:
            similarity = lev_ratio(text, drug_name)
            if (similarity > best_similarity) and (similarity >= self.thresh_text_similarity):
                best_similarity = similarity
                best_similarity_name = drug_name
        if best_similarity_name:
            return True
        
        return False
    
    def map_ocr_results2id_drug(self, ocr_results, text_box_list):
        result_list = []
        with open(self.mapping_drug_name_path) as f:
            mapping_dict = json.load(f)
        drugname_list = mapping_dict.keys()
        for i, text in enumerate(ocr_results):
            text = text.strip()
            best_similarity = 0
            best_similarity_name = None
            for drug_name in drugname_list:
                similarity = lev_ratio(text, drug_name)
                if (similarity > best_similarity) and (similarity >= self.thresh_text_similarity):
                    best_similarity = similarity
                    best_similarity_name = drug_name
            if best_similarity_name:
                id_list = mapping_dict[best_similarity_name]
                id_list = [int(idx) for idx in id_list]
                
                text_info = {"id": id_list, "text": text, "label": "drugname", 
                             "drug_name": best_similarity_name, "box":text_box_list[i]}
                
                result_list.append(text_info)
        
        return result_list