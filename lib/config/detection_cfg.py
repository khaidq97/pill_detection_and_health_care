from ..helpers.config import config

DETECTION_MODELS = {
    'yolov5_detector_model': 0
}
detection_model = config.get('DETECTION_MODEL', cast=int,
                    default=DETECTION_MODELS['yolov5_detector_model'])

# Only for yolov5 
yolov5_detector_model_path = config.get('YOLOV5_DETECTOR_MODEL_PATH', cast=str,
            default=None)