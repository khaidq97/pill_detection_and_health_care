from ..helpers.config import config

model_name = config.get('MODEL NAME', default="vgg_transformer", cast=str)
model_path = config.get('MODEL PATH', default="trained_models/ocr/transformerocr_vietocr_all.pth", cast=str)
mapping_drug_name_path = config.get('MAPPING DRUG NAME TO ID PATH', default="trained_models/drugname_to_id.json", cast=str)
