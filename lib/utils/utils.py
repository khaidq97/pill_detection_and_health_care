import json
import cv2
import numpy as np
from pathlib import Path 
import torch
from Levenshtein import ratio as lev_ratio

# def map_pillname_to_presname(pill_pres_map):
#     with open(str(pill_pres_map)) as f:
#         data = json.load(f)
#     mapping = {}
#     for d in data:
#         pres = d['pres']
#         for d_ in d['pill']:
#             mapping[d_] = pres 
#     return mapping

def map_pillname_to_presname(pill_pres_map):
    with open(str(pill_pres_map)) as f:
        data = json.load(f)
    mapping = {}
    for d in data.keys():
        pres = d 
        for d_ in data[d]:
            d_save = Path(d_).stem
            mapping[d_save] = pres 
    return mapping

def read_json_file(jsonfile):
    with open(str(jsonfile)) as f:
        data = json.load(f)
    return data

def xywh2xyxy(bbox):
    '''
    COCO format to voc format
    '''
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def yolo2xywh(bbox, w, h):
    """
    yolo to coco format
    """
    xmid, ymid = bbox[0]*w, bbox[1]*h 
    w_, h_ = bbox[2]*w, bbox[3]*h 
    xmin = xmid - w_ / 2
    ymin = ymid - h_ / 2
    bbox[0] = xmin
    bbox[1] = ymin 
    bbox[2] = w_ 
    bbox[3] = h_ 
    return bbox

def xyxy2xywh(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    w = xmax - xmin 
    h = ymax - ymin
    bbox[2] = w
    bbox[3] = h
    return bbox


def crop_image(image, box):
    h, w = image.shape[:2]
    # if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
    #     raise ValueError("Cropping Box is not valid")
    # if box[0] > w or box[1] > h or box[2] > w or box[3] > h:
    #     raise ValueError("Cropping Box is not valid")
    # if box[0] > box[2] or box[1] > box[3]:
    #     raise ValueError("Cropping Box is not valid")
    return image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

def draw_bboxes(img, bboxes, texts=None, 
                    bbox_thickness=5,
                    txt_thickness=3,
                    txt_size=1,
                    bbox_color=(114,104,72),
                    txt_color=(26,87,247)):
    if not isinstance(type(bboxes), np.ndarray):
        bboxes = np.array(bboxes)
    bboxes = bboxes.astype(int)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cv2.rectangle(img, bbox[:2], bbox[2:], bbox_color, bbox_thickness)
        if texts is not None:
            x0 = bbox[0]
            y0 = bbox[1]
            text = str(texts[i])
            cv2.putText(img, text, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, txt_thickness)
    return img

def show_pill_label(imgfile, lbfile, size=(1000,1000)):
    img = cv2.imread(str(imgfile))
    bboxes, labels = [], []
    for data in read_json_file(lbfile):
        bbox = [data['x'], data['y'], data['w'], data['h']]
        label = data['label']
        bbox = xywh2xyxy(bbox)
        bboxes += [bbox]
        labels += [label]
    
    img = draw_bboxes(img, bboxes, labels)
    cv2.imshow("PILL Image", cv2.resize(img, size))
    cv2.waitKey()
    cv2.destroyAllWindows()


def pills_to_pres_name(pillfolder, presfolder):
    pillnames = [x.name for x in Path(pillfolder).glob('*') if x.suffix in ('.jpg', '.png')]
    presnames = [x.name for x in Path(presfolder).glob('*') if x.suffix in ('.jpg', '.png')]
    result = {}
    for pillname in pillnames:
        pill_id = pillname.split('_')[2]
        for presname in presnames:
            pres_id = presname.split('_')[-1][:-4]
            if pill_id == pres_id:
                result[pillname] = presname 
                break 
    return result


def draw_pill_groundtruh(im, lbfile):
    bboxes, texts = [], []
    for data in read_json_file(str(lbfile)):
        bbox = [data['x'], data['y'], data['w'], data['h']]
        label = data['label']
        bbox = xywh2xyxy(bbox)
        bboxes += [bbox]
        texts += [label]

    im = draw_bboxes(im, bboxes, texts,
                    bbox_thickness=5,
                    txt_thickness=7,
                    txt_size=3)
    return im

import re
import unicodedata
 
def strip_accents(text):
    """
    Strip accents from input String.
 
    :param text: The input string.
    :type text: String.
 
    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)
 
def text_to_id(text):
    """
    Convert input text to id.
 
    :param text: The input string.
    :type text: String.
 
    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    # text = re.sub('[ ]+', '_', text)
    text = re.sub('[^0-9a-zA-Z_]', '', text)
    return text

def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    inter_w = max(0.0, inter_box[2] - inter_box[0] + 1)
    inter_h = max(0.0, inter_box[3] - inter_box[1] + 1)

    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_area = inter_w * inter_h
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area) 

def extract_id_diagnose(diagnose_text):
  diagnose_id = []
  rule = r"(5|1|\(|[a-z]|[A-Z])([A-Z]|\d)(\d|\.|\?|[A-Z]|\))*" 
  matches = re.compile(rule)
  for i in matches.finditer(diagnose_text):
    diagnose_id.append(i.group())

  return diagnose_id


def processing_diagnose(diagnose):
    if type(diagnose) is str:
        # diagnose = diagnose[2:-2]
        diagnose = diagnose.replace("\n", " ")
        diagnose = diagnose.replace("Chấn đoán: ", "")
        diagnose = diagnose.replace("Chẩn đoán: ", "")
        diagnose = diagnose.replace("Chần đoán: ", "")
        diagnose = diagnose.replace("mạchmáu", "mạch máu")
        diagnose = diagnose.replace("Hộichứng", "Hội chứng")
        diagnose = diagnose.replace("trùngvà", "trùng và")
        diagnose = diagnose.replace("cơđịa", "cơ địa")
        diagnose = diagnose.replace("vaitrái", "vai trái")
        diagnose = diagnose.replace("khôngphân", "không phân")
        diagnose = diagnose.replace("thươngnông", "thương nông")
        diagnose = diagnose.replace("chuyểnhoá", "chuyển hoá")
        diagnose = diagnose.replace("đóiViêm", "đói Viêm")
        diagnose = diagnose.replace("nguyênnhân", "nguyên nhân")
        diagnose = diagnose.replace("bệnhmạch", "bệnh mạch")
        diagnose = diagnose.replace("nãotrong", "não trong")
        diagnose = diagnose.replace("mạchnão", "mạch não")
        diagnose = diagnose.replace("Tràndịch", "Tràn dịch")
        diagnose = diagnose.replace("đặchiệu", "đặc hiệu")
        diagnose = diagnose.replace("khớpvà", "khớp và")
        diagnose = diagnose.replace("môdưới", "mô dưới")
        diagnose = diagnose.replace("hóalipoprotein", "hóa lipoprotein")
        diagnose = diagnose.replace("(nguyênphát)", "(nguyên phát)")
        diagnose = diagnose.replace("Viêmhọng", "Viêm họng")
        diagnose = diagnose.replace("dịứng", "dị ứng")
        diagnose = diagnose.replace("môbào", "mô bào")
        diagnose = diagnose.replace("chânT", "chân")
        diagnose = diagnose.replace("do thiếu", "dothiếu")
        diagnose = diagnose.replace("máunão", "máu não")
        diagnose = diagnose.replace("mềmvùng", "mềm vùng")
        diagnose = diagnose.replace("trongbệnh", "trong bệnh")
        diagnose = diagnose.replace("vàđau", "và đau")
        diagnose = diagnose.replace("(N18)Suy", "(N18) Suy")
        diagnose = diagnose.replace("xácđịnh", "xác định")
        diagnose = diagnose.replace("ÁnChân", "Án Chân")
        diagnose = diagnose.replace(";", " </s>")
        diagnose = diagnose.replace(":", " </s>")
        return diagnose
    else:
        return diagnose


def to_correct_str(b):
  a = re.findall("\(M.*?\)", b)
  d = a.copy()
  for i in range(len(a)):
    if '_' in a[i]:
      a[i] = a[i].replace('_', '-')
  for i in range(len(a)):
    b = b.replace(d[i], a[i])
  b = re.sub(r"[()]", "", b)
  c = b.split('_')
  for i in range(len(c)):
    if c[i] == '160':
      c[i] = c[i].replace('160', '160-1677')
      continue
    if c[i] == '167':
      c[i] = c[i].replace('167', '160-1677')
      continue
    if c[i] == '1677':
      c[i] = c[i].replace('1677', '160-1677')
      continue
    

  return '_'.join(list(set(c))) 


# Hàm tạo ra bert features
@torch.no_grad()
def make_bert_features(text, tokenizer, phobert, max_len=120):
    token_input = tokenizer.encode(text, add_special_tokens=False)
    mask = [1] * len(token_input)
    token_type_ids = [0] * len(token_input)

    padding_len_input = max_len - len(token_input)
    input_ids = token_input + ([0] * padding_len_input)
    attention_mask = mask + ([0] * padding_len_input)
    token_type_ids = token_type_ids + ([0] * padding_len_input)

    input_ids = torch.tensor(input_ids).view(1,-1)
    attention_mask = torch.tensor(attention_mask).view(1,-1)
    token_type_ids = torch.tensor(token_type_ids).view(1,-1)
   
    last_hidden_states,_ = phobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict = False
            )

    return last_hidden_states


# Hàm load model BERT
def load_bert():
    from transformers import  AutoTokenizer, AutoModel
    
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return phobert, tokenizer


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
    
    return 
