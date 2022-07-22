from ..config import app as app_config
from ..config import detection_cfg as detection_config

from .yolov5_detector import yolov5s as yolov5Detector

class DetectionEngine(object):
    def __init__(self, app_config=app_config, detection_config=detection_config):
        detection_model = detection_config.detection_model
        
        if detection_model == detection_config.DETECTION_MODELS['yolov5_detector_model']:
            self.detection_model = yolov5Detector.YoloV5s(
                                device=app_config.device,
                                weight=detection_config.yolov5_detector_model_path
            )

    def predict(self, img):
        bboxes, _ , scores, _  = self.detection_model.predict(img)
        return bboxes, scores