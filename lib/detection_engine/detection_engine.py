from ..config import app as app_config
from ..config import detection_cfg as detection_config

# from .yolov5_detector import yolov5s as yolov5Detector
from .yolov5_detector import yolov5m as yolov5Detector
# from .yolov5_detector import yolov5l as yolov5Detector

class DetectionEngine(object):
    def __init__(self, app_config=app_config, detection_config=detection_config):
        detection_model = detection_config.detection_model
        
        if detection_model == detection_config.DETECTION_MODELS['yolov5_detector_model']:
            print("Yolov5 Detection Engine: ", detection_config.yolov5_detector_model_path)
            self.detection_model = yolov5Detector.YoloV5(
                                device=app_config.device,
                                weight=detection_config.yolov5_detector_model_path
            )

    def predict(self, img):
        bboxes, labels , scores, names  = self.detection_model.predict(img)
        return bboxes, names, scores