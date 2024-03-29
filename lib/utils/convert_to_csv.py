import os
from pathlib import Path
import pandas as pd
import json
import cv2 
from tqdm import tqdm
from lib.utils import utils

class ConvertDataToCSV():
    def __init__(self) -> None:
        pass
    
    def _run_convert(self, pill_label_path, pill_image_path, pres_label_path, save_dir):
        root_dir = save_dir
        train_test_path = ''
        # pill_label_path = os.path.join(root_dir, "pill", "label_cascadercnn_res50")
        # pill_image_path = os.path.join(root_dir, "pill", "image")
        # pres_label_path = os.path.join(root_dir, "prescription", "label")
        # train_test_path = os.path.join(root_dir, "train_val.txt")

        pres_data, pres_doctor, pres_date = self._get_information_in_prescription(pres_label_path)
        
        # image_name = []
        # diagnose = []
        # drugname = []
        # usage = []
        # quantity = []
        # doctor = []
        # date = []
        # label = []
        # for key, value in pres_data.items():
        #     print(value)
        #     image_name.append(key)
        #     diagnose.append(value[0][1][3])
        #     drugname.append("[SEP]".join([i[1][0] for i in value]))
        #     usage.append([i[1][1] for i in value])
        #     quantity.append("[SEP]".join([i[1][2] for i in value]))
        #     doctor.append(pres_doctor[key])
        #     date.append(pres_date[key])
        #     label.append("[SEP]".join([i[1][4] for i in value]))

        # pres_df = pd.DataFrame()
        # pres_df["image_name"] = image_name
        # pres_df["diagnose"] = diagnose
        # pres_df["drugname"] = drugname
        # pres_df["usage"] = usage
        # pres_df["quantity"] = quantity
        # pres_df["doctor"] = doctor
        # pres_df["date"] = date
        # pres_df["label"] = label
        
        # pres_df.to_csv("prescription_test.csv", index=False)
        
        total_data = self._get_information_in_pill(root_dir, pill_label_path, pres_data, pres_doctor, pres_date, train_test_path, pill_image_path)
        df = pd.DataFrame(total_data)

        return df

        # df.to_csv(os.path.join("raw_csv","public_test_cascade"+".csv"), index=False)


    def _get_information_in_prescription(self, pres_label_path):
        pres_label_list = os.listdir(pres_label_path)
        dict_prescription = {}
        doctor_dict = {}
        date_dict = {}
        
        for json_file in pres_label_list:
            json_path = os.path.join(pres_label_path, json_file)
            pres_label = self._load_json(json_path)

            doctor = "None"
            date = "None"
            data_dict = {}
            # data_dict = []
            check = False
            diagnose = ""
            for idx, value in enumerate(pres_label):
                is_first = True
                if value["label"]=="drugname":
                    is_drug=idx
                    check = True
                    usage = []
                    quantity = "None"
                    is_first = False
                    id_mapping = str(value["mapping"])
                if value["label"]=="usage" and is_drug<idx:
                    usage.append(value["text"])
                if (value["label"]=="quantity" and is_first==True) or (value["label"]=="quantity" and is_drug<idx):
                    quantity = value["text"]
                if value['label']=="diagnose":
                    diagnose =  diagnose + value["text"]
                if value["label"]=="date":
                    date = value["text"]
                if "BS. " in value["text"] or "YS." in value["text"]:
                    doctor = value["text"]

                if check:
                    data_dict[pres_label[is_drug]["text"]] = [pres_label[is_drug]["text"],usage,quantity,diagnose, id_mapping]

            dict_prescription[json_file.replace(".json", "")] = [i for i in data_dict.items()]
            doctor_dict[json_file.replace(".json", "")] = doctor
            date_dict[json_file.replace(".json", "")] = date
        return dict_prescription, doctor_dict, date_dict

    def _get_information_in_pill(self, root_dir, pill_label_path, pres_data, pres_doctor, pres_date, train_test_path, pill_image_path):
        pill_label_list = os.listdir(pill_label_path)
        data = []
        count = 0

        pill_pres_map = str(pill_image_path).replace('pill/image', 'pill_pres_map.json')
        pill_to_pres = utils.map_pillname_to_presname(pill_pres_map)

        if os.path.exists(train_test_path):
            train_val_dict = self._get_train_test_split(train_test_path)
       
        for json_file in tqdm(pill_label_list):
            dict_single_pill = {}
            json_path = os.path.join(pill_label_path, json_file)
            path_image = pill_to_pres[Path(json_file).stem]

            pill_label = self._load_json(json_path)

            for idx, value in enumerate(pill_label):
                count+=1
                dict_single_pill['image_name'] = json_file.split('.')[0] + "_" + str(idx) + '.jpg'
                if os.path.exists(train_test_path):
                    dict_single_pill['train/val'] = train_val_dict[json_file.split('.')[0] + '.jpg']
                try:
                    dict_single_pill['id'] = value['label']
                except:
                    pass
                dict_single_pill['bbox'] = str(value['x']) + ' ' + str(value['y']) + ' ' + str(value['w']) + ' ' + str(value['h'])
                if True:
                # try:
                    # print([i[1][1] for i in pres_data[path_image]])
                    # print([i[1][2] for i in pres_data[path_image]])
                    # print(pres_data[path_image])
                    dict_single_pill['diagnose'] = pres_data[path_image][0][1][3]
                    dict_single_pill['drugname'] = "[SEP]".join([i[1][0] for i in pres_data[path_image]])
                    dict_single_pill['usage'] = [i[1][1] for i in pres_data[path_image]]
                    dict_single_pill['SL'] = "[SEP]".join([i[1][2] for i in pres_data[path_image]])
                    dict_single_pill['doctor'] = pres_doctor[path_image]
                    dict_single_pill['date'] = pres_date[path_image]
                # except:
                #     dict_single_pill['diagnose'] = None
                #     dict_single_pill['drugname'] = 'thuoc ngoai'
                #     dict_single_pill['usage'] = None
                #     dict_single_pill['SL'] = None
                #     dict_single_pill['doctor'] = None
                #     dict_single_pill['date'] = None
                data.append(dict_single_pill)
                if os.path.exists(train_test_path):
                    mode = dict_single_pill['train/val']
                else:
                    mode = "test_cascade"
                crop_name = dict_single_pill['image_name']
                dict_single_pill = {}
                
                # save
                origin_name = json_file.split('.')[0]+ '.jpg'
                if not Path(os.path.join(pill_image_path, origin_name)).exists():
                    origin_name = json_file.split('.')[0]+ '.JPG'
                image = cv2.imread(os.path.join(pill_image_path, origin_name))
                x1, y1, x2, y2 = int(value['x']), int(value['y']), int(value['x'])+int(value['w']), int(value['y'])+int(value['h'])
                crop_image = image[y1:y2, x1:x2]
                
                # train_dir = os.path.join(root_dir, "train")
                # val_dir = os.path.join(root_dir, "val")
                # if os.path.exists(train_dir) is False:
                #     os.mkdir(train_dir)
                #     os.mkdir(val_dir)

                save_dir = os.path.join(root_dir, mode)
                if os.path.exists(save_dir) is False:
                    os.mkdir(save_dir)
                cv2.imwrite(os.path.join(save_dir, crop_name), crop_image)
        data = sorted(data, key=lambda x: x['image_name'], reverse = False)

        print(count)
        return data

    def _get_train_test_split(self, train_test_path):
        train_val_dict = {}

        data = open(train_test_path, "r")
        for value in data:
            train_val = value.split(' ')[0]
            name = value.split(' ')[1]
            name = name[:len(name) - 1]
            train_val_dict[name] = train_val

        return train_val_dict
            
    def _load_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return data 


if __name__ == "__main__":
    root_dir = "public_test"
    cvt_data = ConvertDataToCSV()._run_convert(root_dir)
