
class TextRecognition(object):
    def __init__(self, app_config, pres_ocr_config):
        text_recognition_model = pres_ocr_config.text_recognition_model
        if text_recognition_model == pres_ocr_config.TEXT_RECOGNITION_MODELS['viet_ocr']:
            print("Text Recognition: viet ocr: ", pres_ocr_config.viet_ocr_model_path)
            from .viet_ocr import viet_ocr
            self.text_recognition_model = viet_ocr.VietOCR(
                device=app_config.device,
                weight=pres_ocr_config.viet_ocr_model_path
            )
    
    def predict(self, im):
        s = self.text_recognition_model.predict(im)
        return s