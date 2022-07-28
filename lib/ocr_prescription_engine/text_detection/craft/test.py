import os
import glob
import cv2
import argparse
from text_detector_dl import TextDetector

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='checkpoints/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    # parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/media/case.kso@kaopiz.local/New Volume1/hiennt/pill_detection/public_test/prescription/image', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='checkpoints/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()

    save_root = "result/"+args.test_folder.split("/")[-2]+"/"+args.test_folder.split("/")[-1]
    print(save_root)
    os.makedirs(save_root, exist_ok=True)
    text_det = TextDetector(args)
    root = args.test_folder
    img_paths = glob.glob(os.path.join(root, "*"))
    print(len(img_paths))
    start = time.time()
    for img_path in img_paths:
        image = cv2.imread(img_path)
        bboxes = text_det.run_detect(image)
        text_det.save_result(bboxes, img_path, save_root)
    r_time = time.time()-start