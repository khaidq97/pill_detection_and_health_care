import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

class VietOCR(object):
    def __init__(self, device, weight):
        self.device = device
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = weight
        config['cnn']['pretrained'] = False
        config['device'] = device
        config['predictor']['beamsearch'] = False
        self.detector = Predictor(config)

    def predict(self, im):
        im = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        s = self.detector.predict(im)
        if s == '03100000099':
            return None
        else:
            return s