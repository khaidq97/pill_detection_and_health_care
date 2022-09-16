import os
import json
from pathlib import Path 

from tqdm import tqdm
import cv2

from .viet_ocr_recognizer import TextRecognizer
from .text_detection.craft.text_detector_dl import TextDetector
from .field_classifier.rule_base_field_classifier import RuleBaseFieldClassifier
from ..config import text_detection_cfg, ocr_config

class OCRPresciptionEngine():
    def __init__(self):
        self.text_recognizer = TextRecognizer(model_name=ocr_config.model_name, model_path=ocr_config.model_path)
        self.text_detector = TextDetector(text_detection_cfg)
        self.field_classifier = RuleBaseFieldClassifier(ocr_config.mapping_drug_name_path)
        
    def run_state(self, image_root, save_root ,json_root=None):
        image_root, save_root = Path(image_root), Path(save_root)
        if not save_root.exists(): os.makedirs(str(save_root))

        image_paths = [x for x in Path(image_root).rglob('*') if x.suffix in ('.jpg', '.png', '.PNG', '.JPG', '.jpeg', '.JPEG')]
        for image_path in tqdm(image_paths):
            image_name = image_path.stem
            image = cv2.imread(str(image_path))
            if json_root:
                json_path = os.path.join(str(json_root), image_name+".json")
            else:
                json_path = None
            save_path = save_root / (image_name+".json")
            bboxes = self.text_detector.run_detect(str(image_path))
            refine_bboxes = []
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                xmin = max(0, xmin-1)
                ymin = max(0, ymin-2)
                xmax = min(image.shape[1], xmax+1)
                ymax = min(image.shape[0], ymax+2)
                refine_bboxes.append([xmin, ymin, xmax, ymax])
            ocr_results, _ = self.text_recognizer.run_ocr(str(image_path), refine_bboxes)
            # classified_ocr_result example: [{"id": 1, "text": 'something', "label": drugname, "box": [0, 0, 0, 0]}, ...]
            classified_ocr_results = self.field_classifier.classify(ocr_results, bboxes, json_path, have_returned_json=True)
            with(open(str(save_path), 'w', encoding='utf8')) as f:
                json.dump(classified_ocr_results, f, indent = 2, ensure_ascii=False)
            drug_infoes = self.field_classifier.map_ocr_results2id_drug(ocr_results, bboxes)
            # return classified_ocr_results, drug_infoes
