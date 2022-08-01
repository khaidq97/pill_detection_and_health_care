import os
import glob
import json
import random
import re

import cv2
from tqdm import tqdm

from Levenshtein import ratio as lev_ratio

def read_json_file(jsonfile):
    with open(str(jsonfile)) as f:
        data = json.load(f)
    return data

def xywh2xyxy(bbox):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def crop_image(image, box):
    return image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

def draw_bboxes(img, bboxes, texts=None):
    color = [random.randint(0,255) for _ in range(3)]
    colors = [color for _ in range(len(bboxes))]

    for i in range(len(bboxes)):
        bbox, color = bboxes[i], colors[i]
        cv2.rectangle(img, bbox[:2], bbox[2:], color, 10)
        if texts is not None:
            x0 = bbox[0]
            y0 = bbox[1]
            text = str(texts[i])
            cv2.putText(img, text, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)
    return img

def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def is_in_line_boxes(box1, box2, thresh_y_diff_1, thresh_y_diff_2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    if abs(y21 - y11) > thresh_y_diff_1:
        return False
    if y22 - y12 > thresh_y_diff_2:
        return False
    
    return True

def find_in_line_boxes(box1, box_list, thresh_y_diff_1, thresh_y_diff_2):
    in_line_boxes = []
    in_line_indexes = []
    for i, box2 in enumerate(box_list):
        if box2 == box1:
            continue
        if is_in_line_boxes(box1, box2, thresh_y_diff_1, thresh_y_diff_2):
            in_line_boxes.append(box2)
            in_line_indexes.append(i)

    return in_line_boxes, in_line_indexes

def text2IDdrug(text, mapping_dict, thresh_text_similarity=0.8):
    id_list = None
    drugname_list = mapping_dict.keys()
    text = text.strip()
    best_similarity = 0
    best_similarity_name = None
    for drug_name in drugname_list:
        similarity = lev_ratio(text, drug_name)
        if (similarity > best_similarity) and (similarity >= thresh_text_similarity):
            best_similarity = similarity
            best_similarity_name = drug_name
    if best_similarity_name:
        id_list = mapping_dict[best_similarity_name]
        id_list = [int(idx) for idx in id_list]     
    return id_list

def get_diagnose_from_json(json_path):
    diagnose_text_list = []
    with open(json_path, 'r') as f:
        json_obj = json.load(f)
        for element in json_obj:
            if element["label"] == "diagnose":
                diagnose_text_list.append(element["text"])
    output_diagnose_text = ' '.join(diagnose_text_list)
    return output_diagnose_text

def get_splitted_diagnose_name(diagnose_text):
    no_accent_diagnose_text = remove_accents(diagnose_text)
    no_accent_diagnose_text = no_accent_diagnose_text.lower()
    split_diagnose_pattern = r'(;|:)\s+'
    diagnose_list = re.split(split_diagnose_pattern, no_accent_diagnose_text)
    diagnose_list = [text.strip().replace("chan doan", "") for text in diagnose_list]
    diagnose_list = [text for text in diagnose_list if (text != '' and text != ':' and text != ';')]
    return diagnose_list

def classify_diagnose(diagnose_list, save_path=None, similarity_thresh=0.8):
    classification_result = {}
    max_key = 0
    for diagnose_text in diagnose_list:
        if len(classification_result.keys()) == 0:
            classification_result[0] = [diagnose_text]
        else:
            match_flag = False
            for key in classification_result:
                if lev_ratio(classification_result[key][0], diagnose_text) >= similarity_thresh:
                    classification_result[key].append(diagnose_text)
                    match_flag = True
                    break
                
            if not match_flag:
                max_key += 1
                classification_result[max_key] = [diagnose_text]
    if save_path:
        with(open(save_path, 'w')) as f:
            json.dump(classification_result, f, indent=2)
    return classification_result

def get_all_diagnose_texts(json_root):
    diagnose_list = []
    json_paths = glob.glob(os.path.join(json_root, "*.json"))
    for json_path in tqdm(json_paths):
        diagnose_text = get_diagnose_from_json(json_path)
        split_diagnose_list = get_splitted_diagnose_name(diagnose_text)
        diagnose_list.extend(split_diagnose_list)
    
    return diagnose_list
        
if __name__ == '__main__':
    json_root = '/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection/public_train/prescription/label'
    diagnose_list = get_all_diagnose_texts(json_root)
    print(diagnose_list)
    save_path = '/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection/public_train/prescription/diagnose_classification.json'
    classification_result = classify_diagnose(diagnose_list, save_path)
        
