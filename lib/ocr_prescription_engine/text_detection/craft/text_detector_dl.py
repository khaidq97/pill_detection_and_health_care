"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from .craft import craft_utils, imgproc
from .craft.craft import CRAFT
from .craft.refinenet import RefineNet
from ..text_detector_ip import *

import time

class TextDetector:

    def __init__(self, cfg):
        # Only load CRAFT if not using predefined result
        self.net = None
        self.refine_net = None
        self.cfg = cfg
        self._load_models()

        self.image_log_dir = None
        self.debug = False

    def run_detect(self, image_path, image_log_dir=None, debug=False):
        start = time.time()
        # resize_image = imutils.resize(image, height=640)
        # ratio = image.shape[0]/resize_image.shape[0]
        image = cv2.imread(image_path)
        resize_image = image

        self.image_log_dir = image_log_dir
        self.debug = debug

        rgb_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        text_boxes = self.detect_text(
            rgb_image,
            0.7,
            0.3,
            0.35,
            True,
            False)
        end = time.time()
        # remove zero-size and correct wrong-coordinate text_boxes
        h, w = resize_image.shape[:2]
        for box in text_boxes:
            box["x1"] = max(0, box["x1"])
            box["y1"] = max(0, box["y1"])
            box["x2"] = max(0, box["x2"])
            box["y2"] = max(0, box["y2"])
            box["x1"] = min(w - 1, box["x1"])
            box["y1"] = min(h - 1, box["y1"])
            box["x2"] = min(w - 1, box["x2"])
            box["y2"] = min(h - 1, box["y2"])
        text_boxes = [box for box in text_boxes
                      if box["x2"] - box["x1"] > 0
                      and box["y2"] - box["y1"] > 16]

        # for box in text_boxes:
        #     box["x1"] = int(box["x1"]*ratio)
        #     box["y1"] = int(box["y1"]*ratio)
        #     box["x2"] = int(box["x2"]*ratio)
        #     box["y2"] = int(box["y2"]*ratio)
        # text_boxes = self.refactor_text_boxes(text_boxes)
        text_boxes = [[text_box['x1'], text_box['y1'], text_box['x2'], text_box['y2']] for text_box in text_boxes]
        return text_boxes

    def refactor_text_boxes(self, text_boxes):
        text_boxes += [text_boxes[-1]]
        text_boxes += [text_boxes[-1]]
        new_text_boxes = []
        skip_text_boxes_id = []
        for i, text_box in enumerate(text_boxes[:-2]):
            if i in skip_text_boxes_id:
                continue
            combine_text_box = None
            text_box_height = text_box['y2'] - text_box['y1']
            text_box_width = text_box['x2'] - text_box['x1']
            for j, next_text_box in enumerate(text_boxes[i+ 1: i + 3]):
                next_text_box_height = next_text_box['y2'] - next_text_box['y1']
                next_text_box_width = next_text_box['x2'] - next_text_box['x1']
                if 0.9 * text_box_height < next_text_box_height < 1.1 * text_box_height:
                    if (text_box['y1'] + text_box_height/3) < (next_text_box['y1'] + next_text_box['y2']) / 2 < (text_box['y2'] - text_box_height/3):
                        if (text_box_width / text_box_height) < 4 and (next_text_box_width / next_text_box_height) < 4:
                            combine_text_box = next_text_box
                            skip_text_boxes_id.append(i + j + 1)
                            break
            if combine_text_box is not None:
                box = {}
                box["x1"] = min(text_box['x1'], combine_text_box["x1"])
                box["y1"] = min(text_box['y1'], combine_text_box["y1"])
                box["x2"] = max(text_box['x2'], combine_text_box["x2"])
                box["y2"] = max(text_box['y2'], combine_text_box["y2"])
                new_text_boxes.append(box)
            else:
                new_text_boxes.append(text_box)

        return new_text_boxes


    def _inference(self, image, text_threshold, link_threshold, low_text, cuda, poly, check_func=np.max, log_dir=None):
        start = time.time()
        image_log_dir = self.image_log_dir
        if self.debug:
            image_log_dir = log_dir
            os.makedirs(image_log_dir, exist_ok=True)

        if not self.net:
            self._load_models()

        # resize
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image, 1280,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

        start_model_time = time.time()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)
        
        end_model_time = time.time()
        # print("Running time MODEL: ", end_model_time-start_model_time)
        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        start_refine_time = time.time()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
            start_pos = time.time()
            # Post-processing
            boxes, polys = craft_utils.getDetBoxes(
                score_text, score_link, text_threshold, link_threshold, low_text, poly=poly, check_func=check_func,
                log_dir=image_log_dir)
            # print("TIME POST PROCESS: ", time.time()-start_pos)
        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k, poly in enumerate(polys):
            if poly is None:
                polys[k] = boxes[k]

        end_refine_time = time.time()
        # print("Running time refine: ", end_refine_time-start_refine_time)
        # render results (optional)
        # print("Running time total: ", time.time()-start)
        if self.debug:
            os.makedirs(image_log_dir, exist_ok=True)
            render_img = score_text.copy()
            render_img = np.hstack((render_img, score_link))
            ret_score_text = imgproc.cvt2HeatmapImg(render_img)
            cv2.imwrite(os.path.join(image_log_dir,
                                     "craft_text_detection.png"), ret_score_text)
            cv2.imwrite(os.path.join(image_log_dir,
                                     "craft_text_detection_text_map.png"), score_text * 255)
            cv2.imwrite(os.path.join(image_log_dir,
                                     "craft_text_detection_link_map.png"), score_link * 255)
        else:
            ret_score_text = None

        return boxes, polys, ret_score_text

    def _copy_state_dict(self, state_dict):
        if list(state_dict)[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def _load_models(self):
        """
        Load CRAFT models
        """

        self.net = CRAFT()  # initialize

        print('Loading trained_models/text_detection/craft/weights from checkpoint (' +
              self.cfg.craft_detection_model_path + ')')
        if False:
            self.net.load_state_dict(self._copy_state_dict(
                torch.load(self.cfg.craft_detection_model_path)))
            # self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        else:
            self.net.load_state_dict(self._copy_state_dict(
                torch.load(self.cfg.craft_detection_model_path, map_location='cpu')))
        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if True:
            self.refine_net = RefineNet()
            print('Loading trained_models/text_detection/craft/weights of refiner from checkpoint (' +
                  self.cfg.craft_refine_model_path + ')')
            if False:
                self.refine_net.load_state_dict(self._copy_state_dict(
                    torch.load(self.cfg.craft_refine_model_path)))
                # self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(self._copy_state_dict(
                    torch.load(self.cfg.craft_refine_model_path, map_location='cpu')))

            self.refine_net.eval()

    def detect_text(self, rgb_image, text_thresh, link_thresh, low_text_thresh, use_gpu, output_poly, check_func=np.max,
                    log_dir=None):
        bboxes, _, _ = self._inference(rgb_image,
                                       text_thresh, link_thresh,
                                       low_text_thresh, use_gpu,
                                       output_poly, check_func=check_func, log_dir=log_dir)
        # convert CRAFT result text_boxes
        text_boxes = [
            {"x1": int(box[0][0]), "y1": int(box[0][1]),
             "x2": int(box[2][0]), "y2": int(box[2][1])}
            for box in bboxes
        ]

        return text_boxes

    def save_result(self, bboxes, image_path, save_root, suffix=""):
        image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1].split(".")[0]
        label_path = os.path.join(save_root, image_name+suffix+".txt")
        with open(label_path, 'w') as fw:
            for box in bboxes:
                image = cv2.rectangle(image, (box['x1'], box['y1']), (box['x2'], box['y2']), (0,0,255), thickness=2)
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                x, y, w, h = self.convert2yolo_format(image, box)
                data = "0 "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n"
                fw.write(data)
        cv2.imwrite(os.path.join(save_root, image_name+".jpg"), image)
    
    def convert2yolo_format(self, image, box):
        h, w, _ = image.shape
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        x = (x1+x2)/(2*w)
        y = (y1+y2)/(2*h)
        w_box = (x2-x1)/w
        h_box = (y2-y1)/h
        return x, y, w_box, h_box