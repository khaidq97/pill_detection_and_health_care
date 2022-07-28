from ..detection_engine import detection_engine
from ..pres_ocr_engine import pres_ocr_engine
from . import post_processing

class Controller(object):
    def __init__(self):
        self.detectionEngine = detection_engine.DetectionEngine()
        self.presOCREngine = pres_ocr_engine.PresOCREngine()
        self.postProcessing = post_processing.PostProcessing()

    def predict(self, pill_im, pres_im):
        pill_bboxes, pill_ids, pill_scores = self.detectionEngine.predict(pill_im.copy())
        pres_result = self.presOCREngine.predict(pres_im.copy())

        # Hard postporcessing
        bboxes, ids, scores = pill_bboxes, pill_ids, pill_scores
        if pill_bboxes is not None:
            ids = self.postProcessing.hard_postprocessing(pill_ids, pres_result)
        
        return bboxes, ids, scores


