import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import os
import json
import cv2

def one_hot_fn(x, num_dim):
    vector = np.zeros(num_dim)
    try:
        vector[x] = 1
        return vector
    except: 
        return vector

def processing_id(id):
    id = id +1
    return id

def processing_drugname(drugname):
    drugnames = drugname.split("[SEP]")
    drugname = [i[3:] for i in drugnames]
    return "[SEP]".join(drugname)

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
        diagnose = diagnose.replace("ÁnChân", "Án Chân") #khôngphụ
        diagnose = diagnose.replace("khôngphụ", "không phụ") #khôngphụ
        diagnose = diagnose.replace("-", "") #khôngphụ
        diagnose = diagnose.replace(";", " </s>")
        diagnose = diagnose.replace(":", " </s>")
        return diagnose.strip()
    else:
        return diagnose
    
def preprocessing_dataframe(df):
    # df = df[df["id"]!=107]
    # df.rename(columns = {'diagnoise':'diagnose'}, inplace = True)
    
    # pres_name_list = []
    # for image_name in df['image_name'].values:
    #     pres_name = image_name.split("_")[0] + "_P_TRAIN_"+ image_name.split("_")[2] + ".png"
    #     pres_name_list.append(pres_name)
    # df["prescription_name"] = pres_name_list
        
    df["diagnose"] = df["diagnose"].apply(lambda x: processing_diagnose(x))
    # df["drugname"] = df["drugname"].apply(lambda x: processing_drugname(x))
    # df["id"] = df["id"].apply(lambda x: processing_id(x))
    # df["diagnose"] = df["diagnose"].fillna("empty")
    # df["SL"] = df["SL"].fillna("empty")
    try:
        train_df = df[df['train/val']=="train"]
        val_df = df[df['train/val']=="val"]
    except:
        train_df = None
        val_df = None
        
    return train_df, val_df, df

def get_drugname_2_id(df):
    id_2_drugname = {}
    drugname_2_id = {}
    drugnames_unique = []
    for idx, drugname in enumerate(df["drugname"].unique()):
        drugs = drugname.split("[SEP]")
        drugnames_unique.extend(drugs)
    for idx, drugname  in enumerate(np.unique(drugnames_unique)):
        id_2_drugname[idx] = drugname
        drugname_2_id[drugname] = idx
        
    return id_2_drugname, drugname_2_id

def get_drugname_2_label(json_path="name2id.json"):
    with open(json_path, "r") as file:
        name2id = json.load(file)
        
    return name2id

def get_dotorname_2_id(df):
    doctor_2_id = {}
    for idx, doctor in enumerate(df["doctor"].unique()):
        doctor_2_id[doctor] = idx
        
    return doctor_2_id
    
    
def except_108(arr):
    out = []
    for i in arr:
        if i!="108":
            out.append(int(i))
            
    return out

def is_empty(arr):
    if len(arr)==0:
        return True
    else:
        return False
    
def cls_metrics(targets, predictions):
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro")

    return acc, f1

def load_checkpoint(checkpoint, model, optimizer = None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    train_f1 = checkpoint['train_f1_score']
    val_f1 = checkpoint['val_f1_score']
    return last_epoch, train_f1, val_f1

def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def padding_image(image, new_size=(320,320)):
    old_h, old_w, c = image.shape
    new_h, new_w= new_size
    
    padding = np.full((new_h,new_w, c), (255,255,255), dtype=np.uint8)
    # compute center offset
    if old_h < old_w:
        w = new_w 
        h = int((new_w * old_h)/old_w)
    else:
        h = new_h
        w = int((new_h * old_w)/old_h)
        
    x_center = (new_w - w) // 2
    y_center = (new_h - h) // 2
        
    new_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
    
    # copy img image into center of result image
    padding[y_center:y_center+h, x_center:x_center+w] = new_image
    
    return padding


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    return data
# img = cv2.imread("test.png")
# img = padding_image(img)
# cv2.imwrite("test_2.png", img)