import os
import pandas as pd

from tools.convert_to_csv import ConvertDataToCSV
from tools.preprocessing_data import *
from lib.classification_engine.cls_module.utils import *
from lib.classification_engine.classification_engine import ClassificationEngine
if __name__ == "__main__":
    # convert to csv
    root_dir = "public_test"
    if os.path.exists(os.path.join(root_dir, "processed_"+root_dir+".csv")) is False:
        dataframe = Cvt_data = ConvertDataToCSV()._run_convert(root_dir=root_dir)
        # preprocessing data
        dataframe["diagnose"] = dataframe["diagnose"].apply(lambda x: processing_diagnose(x))
        dataframe["drugname"] = dataframe["drugname"].apply(lambda x: processing_drugname(x))
        # dataframe.to_csv("test.csv", index=False)


        dataframe = encode_druname(dataframe)
        # dataframe = encode_usage(dataframe)
        dataframe = encode_quantity(dataframe)
        dataframe = encode_doctor(dataframe)
        dataframe = mapping_druname_to_label(dataframe)

        # saving processed dataframe
        dataframe.to_csv(os.path.join(root_dir, "processed_"+root_dir+".csv"), index=False)

    else:
        dataframe = pd.read_csv(os.path.join(root_dir, "processed_"+root_dir+".csv"))

    folder = os.path.join(root_dir, "test_cascade")
    ckpt_dir = "trained_models/classification_module/swin_True_True_True_tiny_v1.pt"
    device = torch.device("cpu")
    drugname_path = "lib/classification_engine/mapping/drugname_2_id.json"
    doctor_path = "lib/classification_engine/mapping/doctor_2_id.json"
    classify_engine = ClassificationEngine(ckpt_dir=ckpt_dir, device=device, drugname_2_id_path=drugname_path, doctor_2_id_path=doctor_path)
    classify_engine.predict_dataframe(dataframe=dataframe, img_folder=folder, saved_dir="log_dir")
    

    
