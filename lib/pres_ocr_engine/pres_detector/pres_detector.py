
class PresDetector(object):
    def __init__(self, app_config, pres_ocr_config):
        pres_detection_model = pres_ocr_config.pres_detection_model
        if pres_detection_model == pres_ocr_config.PRES_DETECTION_MODELS['pres_yolov5_detector']:
            print("pres_yolov5_detector: ", pres_ocr_config.pres_yolov5_model_path)
            from .pres_yolov5_detector import yolov5s as yolov5Detector
            self.pres_detection_model = yolov5Detector.YoloV5(
                device=app_config.device,
                weight=pres_ocr_config.pres_yolov5_model_path
            )
    
    def predict(self, im):
        bboxes, labels, scores, names = self.pres_detection_model.predict(im)
        return bboxes, names, scores