import os
from lib.ocr_prescription_engine.ocr_prescription_engine import OCRPresciptionEngine

ocr = OCRPresciptionEngine()
image_root = "/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection/public_train/prescription/image"
save_root = "/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection/public_train/prescription/ocr_run_label"
json_root = None
os.makedirs(save_root, exist_ok=True)
ocr.run_ocr(image_root, json_root, save_root)