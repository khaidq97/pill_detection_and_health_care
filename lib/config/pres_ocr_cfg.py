from ..helpers.config import config

#=========================For pres detector=============================#
PRES_DETECTION_MODELS = {
    'pres_yolov5_detector': 0
}
pres_detection_model = config.get('PRES_DETECTION_MODEL', cast=int,
                        default=PRES_DETECTION_MODELS['pres_yolov5_detector'])

# Only for yolov5
pres_yolov5_model_path = config.get('PRES_YOLOV5_MODEL_PATH', cast=str, default=None)

#========================For tex Recognition==============================#
TEXT_RECOGNITION_MODELS = {
    'viet_ocr': 0
}
text_recognition_model = config.get('TEXT_RECOGNITION_MODEL', cast=int,
                       default=TEXT_RECOGNITION_MODELS['viet_ocr'])
# Only for viet_ocr
viet_ocr_model_path = config.get('VIET_OCR_MODEL_PATH', cast=str, default=None)

#========================For Post process=================================#
drugname_to_ids = config.get('DRUGNAME_TO_IDS', cast=str, default=None)

