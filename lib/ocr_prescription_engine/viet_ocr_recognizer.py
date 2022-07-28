import os
import glob
import argparse

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import sys
from PIL import Image

from ..utils.utils import remove_accents

class TextRecognizer():
    def __init__(self, model_name='vgg_seq2seq', model_path=None, device='cpu', have_beamsearch=False, cnn_pretrained=False):
        self.config = Cfg.load_config_from_name(model_name)
        if model_path:
            self.config['weights'] = model_path
        self.config['device'] = device
        self.config['cnn']['pretrained'] = cnn_pretrained
        self.config['predictor']['beamsearch'] = have_beamsearch
        self.model = Predictor(self.config)

    def run_ocr(self, image_path, bboxes, debug=False, debug_dir=""):
        image = cv2.imread(image_path)
        ocr_results = []
        for text_box in bboxes:
            x1, y1, x2, y2 = text_box
            text_image = image[y1:y2, x1:x2]
            text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2RGB)
            text_image = Image.fromarray(text_image)
            text = self.model.predict(text_image)
            ocr_results.append(text)

        if debug:
            image_name = image_path.split("/")[-1].split(".")[0]
            for i, text in enumerate(ocr_results):
                text_box = bboxes[i]
                x1, y1, x2, y2 = text_box
                text_image = image[y1:y2, x1:x2]
                text = remove_accents(text)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0,0,255), thickness=2)
                image = cv2.putText(image, text, org=(x1, max(y1-5, 0)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale=1, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
                cv2.imwrite(os.path.join(debug_dir, image_name+".jpg"), image)
        
        return ocr_results, bboxes
