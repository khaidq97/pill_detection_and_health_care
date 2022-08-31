import os 

from lib.detection_engine import detection_engine
from lib.ocr_prescription_engine import ocr_prescription_engine
from lib.classification_engine import classification_engine

class Controller(object):
    def __init__(self):
        pass

    def run_state(self, pill_dir, pres_dir, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        pill_image_path = pill_dir
        pres_image_path = pres_dir
        pill_label_path = os.path.join(save_dir, 'pill_label_path')
        pres_label_path = os.path.join(save_dir, 'pres_label_path')

        print('\nRUNINIG ON BATCH MODE...')
        print('\nPill Detection state...')
        self.detectionEngine = detection_engine.DetectionEngine()
        self.detectionEngine.run_state(pill_image_path, pill_label_path)
        del self.detectionEngine

        print('\nOCR prescription state...')
        self.ocrpresEngine = ocr_prescription_engine.OCRPresciptionEngine()
        self.ocrpresEngine.run_state(pres_image_path, pres_label_path)
        del self.ocrpresEngine

        print('\nClassification state...')
        print(f'Save at: {save_dir}/results.csv')
        self.classificationEngine = classification_engine.ClassificationEngine()
        self.classificationEngine.run_state(pill_label_path, pill_image_path, pres_label_path, save_dir)
