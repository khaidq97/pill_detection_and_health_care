from lib.ocr_prescription_engine.ocr_prescription_engine import OCRPresciptionEngine

ocr = OCRPresciptionEngine()
image_path = "/media/case.kso@kaopiz.local/New Volume1/hiennt/pill_detection/public_train/prescription/image/VAIPE_P_TRAIN_1.png"
ocr.run_ocr(image_path)