from pydoc import doc
import pandas as pd
import numpy as np
import os
import json
import re
from sklearn.preprocessing import MinMaxScaler
from difflib import SequenceMatcher
from Levenshtein import ratio as lev_ratio

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
    return best_similarity_name

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    return data

def correct_drugname(drugname):
    drugname = drugname.replace("DROXICEF 500MG 500mg", "DROXICEF 500MG 0,5g")
    drugname = drugname.replace("HOẠT HUYẾT DƯỠNG NÃO BDF 150mg + 5mg", "HOẠT HUYẾT DƯỠNG NÃO BDF 150mg+5mg")
    drugname = drugname.replace("LIVONIC 2500mg+400mg +500mg 485mg", "LIVONIC 2500mg+400mg +500mg +85mg")
    drugname = drugname.replace("TIOGA 33,333mg+1g+0,34g+0,25g+0,17g", "TIOGA 33,33mg+1g+0,34g+0,25g+0,17g")
    
    return drugname

def processing_drugname(drugname):
    drugname = correct_drugname(drugname)
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
        diagnose = diagnose.replace("khôngphụ", "không phụ")
        diagnose = diagnose.replace("[10", "I10")
        diagnose = diagnose.replace("110", "I10")
        diagnose = diagnose.replace("JII", "J11")
        diagnose = diagnose.replace("[31", "J31")
        diagnose = diagnose.replace("Ell", "E11")
        diagnose = diagnose.replace("tìnhtrạng", "tình trạng")
        diagnose = diagnose.replace("tỉnhtrạng", "tình trạng") #khôngđặc
        diagnose = diagnose.replace("mềmngực", "mềm ngực")
        diagnose = diagnose.replace("khôngđặc", "không đặc")
        diagnose = diagnose.replace("-", "") #khôngphụ
        diagnose = diagnose.replace(";", " </s>")
        diagnose = diagnose.replace(":", " </s>")
        return " ".join(diagnose.split())
    else:
        return diagnose

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

def mapping_druname_to_label(df, name2id):
    encode_drugname_list = []
    for drugnames in df['drugname'].values:
        drugs = drugnames.split("[SEP]")
        drugs_in_pres = []
        for drug in drugs:
            try:
                drugs_in_pres.extend(name2id[drug])
            except:
                drugs_in_pres.append(108)
        encode_drugname_list.append(' '.join(str(x) for x in drugs_in_pres))
        
    return encode_drugname_list

def encode_druname(df, max_len=5):
    drug_name_list = load_json("drugname_2_id.json")
    encode_drugname_list = []
    for drugnames in df['drugname'].values:
        drugs = drugnames.split("[SEP]")
        drugs_in_pres = []
        for drug in drugs:
            drug = text2IDdrug(drug, drug_name_list)
            drugs_in_pres.append(int(drug_name_list[drug]))
        padding_len = max_len - len(drugs_in_pres)
        drugs_in_pres = drugs_in_pres + [0] * padding_len
        drugs_in_pres = np.array(drugs_in_pres) / 142
        encode_drugname_list.append(' '.join(str(x) for x in drugs_in_pres))
        # encode_drugname_list.append(drugs_in_pres)
    
    df["encode_drugname"] = encode_drugname_list
    
    return df

def get_usage_each(text):
    out = []
    rule = r"\[.*?\]"
    matches = re.compile(rule)
    for match in re.finditer(matches, text):
        out.append(match.group())
    return out

def encode_usage(df, max_len=5):
    encode_usage_list = []
    for usage in df['usage'].values:
        usage = usage[1:-1]
        usage_list = get_usage_each(usage)
        
        usage_list_drug = []
        for i, u in enumerate(usage_list):
            u = u[1:-1].replace("'","").split(", ")
            if u[0]=="":
                u = "empty"
            else: 
                u = ", ".join(u)
            usage_list_drug.append(u)
        usage_list_drug = "[SEP]".join(usage_list_drug)
        encode_usage_list.append(usage_list_drug)
        
    df["usage"] = encode_usage_list
    
    return df

def encode_quantity(df, max_len=5):
    encode_quantity_list = []
    for drugnames in df['SL'].values:
        drugs = drugnames.split("[SEP]")
        drugs_in_pres = []
        for drug in drugs:
            match = re.search(r"\d.?", drug)
            if match:
                number = match.group()
                drugs_in_pres.append(int(number))
            else:
                drugs_in_pres.append(0)
        padding_len = max_len - len(drugs_in_pres)
        drugs_in_pres = drugs_in_pres + [0] * padding_len
        drugs_in_pres = np.array(drugs_in_pres) / 100
        encode_quantity_list.append(' '.join(str(x) for x in drugs_in_pres))
        # break
    df["encode_quantity"] = encode_quantity_list
    # print(df["encode_quantity"])
    return df


def get_dotorname_2_id(df):
    doctor_2_id = {}
    for idx, doctor in enumerate(df["doctor"].unique()):
        if "BS. "in doctor or "YS. " in doctor:
            doctor_2_id[doctor] = idx+1
        
    return doctor_2_id

def encode_doctor(df):
    doctor_2_id = load_json("doctor_2_id.json")
    # print(doctor_2_id)
    encode_doctor_list = []
    for doctor in df["doctor"].values:
        if doctor in doctor_2_id.keys():
            encode_doctor_list.append(doctor_2_id[doctor])
        else:
            encode_doctor_list.append(0)
            
        
    df["encode_doctor"] = np.array(encode_doctor_list) / 69

    return df


def encode_date(df):
    day_list, month_list, year_list = [], [], []
    
    for date in df["date"].values:
        rule = r"\d{2}\/\d{2}\/\d{4}"
        match = re.search(rule, date)
        if match:
            date = match.group()
            day_list.append(date.split("/")[0])
            month_list.append(date.split("/")[1])
            year_list.append(date.split("/")[2])
        else:
            date = "empty"
            day_list.append(0)
            month_list.append(0)
            year_list.append(0)
            
    # df["day"] = np.array(day_list) / 31
    # df["month"] = np.array(month_list) / 12
    # df["year"] = np.array(year_list) / 2022
    df["day"] = np.array(day_list)
    df["month"] = np.array(month_list)
    df["year"] = np.array(year_list) 

    return df


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def correct_diagnose_id(id):
    diagnose_2_id = load_json("diagnose_2_id.json")
    diagnose_ids = diagnose_2_id.keys()
    max = 0
    max_diagnose_id = "None"
    for i, diagnose_id in enumerate(diagnose_ids):
        score = similar(id, diagnose_id)
        if score > max:
            max = score
            max_diagnose_id = diagnose_id
    
    return max_diagnose_id


def encode_diagnose(df):
    encode_diagnose_list = []
    for diagnose in df["diagnose"].values:
        ids = diagnose.split("</s>")
        id_list = []
        for id in ids:
            id = id.split()[0]
            id = correct_diagnose_id(id)
            id_list.append(id)
        id_list = "_".join(id_list)
        encode_diagnose_list.append(id_list)
        
    df["encode_diagnose"] = encode_diagnose_list
    out = df[["image_name","encode_diagnose"]] 
    out.to_csv("test.csv", index=False)
    return out

def add_image_name(text):
    return "VAIPE_P_TEST_"+str(text)
    
if __name__ == "__main__":
    file_name = "public_train_original"
    df = pd.read_csv(os.path.join("raw_csv_v3", file_name+".csv"))
    
    with open("name2id_v2.json", "r") as file:
        name2id = json.load(file)
    
    df["diagnose"] = df["diagnose"].apply(lambda x: processing_diagnose(x))
    df["drugname"] = df["drugname"].apply(lambda x: processing_drugname(x))
    
    df = encode_druname(df)
    df = encode_usage(df)
    df = encode_quantity(df)
    # df = encode_doctor(df)
    # df = encode_date(df)
    mapping = mapping_druname_to_label(df, name2id)
    df['prescription_mapping'] = mapping
    
    # df.to_csv(os.path.join("prescription_csv", file_name+"_encoded.csv"), index=False)
    df.to_csv(os.path.join("classification_model", "pill_csv_fix_bbox", file_name+"_encoded.csv"), index=False)