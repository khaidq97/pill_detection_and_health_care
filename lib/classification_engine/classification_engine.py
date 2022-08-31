from ..config import app
from ..config import classification_cfg 

from lib.utils.convert_to_csv import ConvertDataToCSV
from lib.utils.preprocessing_data import *
from lib.classification_engine.cls_module.utils import *
from lib.classification_engine import classification

class ClassificationEngine(object):
    def __init__(self, drugname_path="lib/classification_engine/mapping/drugname_2_id.json",
                       doctor_path= "lib/classification_engine/mapping/doctor_2_id.json"):
        self.classify_engine = classification.ClassificationEngine(ckpt_dir=classification_cfg.classification_engine_model_path, 
                                        device=app.device, 
                                        drugname_2_id_path=drugname_path, 
                                        doctor_2_id_path=doctor_path)

    def _gen_csv(self, pill_label_path, pill_image_path, pres_label_path, save_dir):
        train_test_path = ''
        dataframe = ConvertDataToCSV()._run_convert(pill_label_path, pill_image_path, pres_label_path, save_dir)
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
        dataframe.to_csv(os.path.join(save_dir, "processed_"+save_dir+".csv"), index=False)
        return dataframe


    def run_state(self, pill_label_path, pill_image_path, pres_label_path, save_dir):
        if os.path.isfile(os.path.join(save_dir, "processed_"+save_dir+".csv")):
            print('Loading csv file ..')
            dataframe = pd.read_csv(os.path.join(save_dir, "processed_"+save_dir+".csv"))
        else:
            print('Creating csv file...')
            dataframe = self._gen_csv(pill_label_path, pill_image_path, pres_label_path, save_dir)

        print('Classifying id ...')
        folder = os.path.join(save_dir, "test_cascade")
        self.classify_engine.predict_dataframe(dataframe=dataframe, img_folder=folder, saved_dir=save_dir)