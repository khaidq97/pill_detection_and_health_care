from .viet_ocr_recognizer import TextRecognizer
from .text_detection.craft.text_detector_dl import TextDetector
from ..classification_engine.field_classifier.rule_base_field_classifier import RuleBaseFieldClassifier
from ..config import text_detection_cfg, ocr_config

class OCRPresciptionEngine():
    def __init__(self):
        self.text_recognizer = TextRecognizer(model_name=ocr_config.model_name, model_path=ocr_config.model_path)
        self.text_detector = TextDetector(text_detection_cfg)
        self.field_classifier = RuleBaseFieldClassifier(ocr_config.mapping_drug_name_path)
        
    def run_ocr(self, image_path):
        bboxes = self.text_detector.run_detect(image_path)
        ocr_results, _ = self.text_recognizer.run_ocr(image_path, bboxes)
        # classified_ocr_result example: [{"id": 1, "text": 'something', "label": drugname, "box": [0, 0, 0, 0]}, ...]
        classified_ocr_results = self.field_classifier.classify(ocr_results, bboxes, have_returned_json=True)
        drug_infoes = self.field_classifier.map_ocr_results2id_drug(ocr_results, bboxes)
        return classified_ocr_results, drug_infoes