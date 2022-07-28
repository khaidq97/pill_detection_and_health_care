from ..config import app 
from ..config import pres_ocr_cfg

from .pres_detector import pres_detector
from .text_recognition import text_recognition
from ..utils import utils
from . import post_process

class PresOCREngine(object):
    def __init__(self, app_config=app, pres_ocr_config=pres_ocr_cfg):
        self.presDetector = pres_detector.PresDetector(
            app_config=app_config,
            pres_ocr_config=pres_ocr_config
        )
        self.textRecognition = text_recognition.TextRecognition(
            app_config=app_config,
            pres_ocr_config=pres_ocr_config
        )

    def predict(self, im):
        h, w = im.shape[:2]
        bboxes, labels, _ = self.presDetector.predict(im)
        
        if bboxes is not None:
            results = []
            for bbox, lb in zip(bboxes, labels):
                try:
                    bbox = self.expand_bbox(bbox, w, h, deltax=10, deltay=4)
                    im_text = utils.crop_image(im, bbox)
                    text = self.textRecognition.predict(im_text)
                    
                    if lb == 'drugname':
                        ids, text = post_process.drugname_postprocess(text)
                    else:
                        ids = None
                    results.append({
                        'text': text,
                        'bbox': bbox,
                        'label': lb,
                        'ids': ids
                    })
                except:
                    continue
            return results
        else:
            return None

    def expand_bbox(self, bbox, w, h, deltax=10, deltay=2):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin -= deltax 
        xmax += deltax
        ymin -= deltay
        ymax += deltay
        if xmin < 0: xmin = 0
        if xmax > w: xmax = w
        if ymin < 0: ymin = 0
        if ymax > h: ymax = h
        bbox[0] = xmin 
        bbox[2] = xmax
        return bbox

        
