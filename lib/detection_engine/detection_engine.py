import mmcv
import numpy as np
from tqdm import tqdm
from pathlib import Path 
import os
import cv2
import json

from .mmdet.apis.inference import init_detector, inference_detector  # ,show_result
from ..config import app, detection_cfg

class DetectionEngine(object):
    def __init__(self, config_file='lib/detection_engine/cascade_rcnn_r50_fpn_1x_coco.py',
                        score_thr=0.8):
        self.score_thr = score_thr
        self.model = init_detector(config_file, detection_cfg.detection_engine_model_path, app.device)

    def run(self, im):
        result = inference_detector(self.model, im.copy())
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        
        if bboxes is not None:
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :4]
            labels = labels[inds]
            scores = scores[inds]
            return bboxes, scores, labels
        else:
            return None, None, None


    def run_state(self, folder, savefolder):
        folder, savefolder = Path(folder),  Path(savefolder)
        if not savefolder.exists(): os.makedirs(str(savefolder))

        imfiles = [x for x in folder.rglob('*') if x.suffix in ('.jpg', '.png', '.PNG', '.JPG', '.jpeg', '.JPEG')]
        for imfile in tqdm(imfiles):
            im = cv2.imread(str(imfile))
            bboxes, scores, _ = self.run(im)
            temp = []
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox.tolist()
                temp.append({
                    'x': float(xmin),
                    'y': float(ymin),
                    'h': float(ymax - ymin),
                    'w': float(xmax - xmin),
                    'label': 'pill'
                })
            savefile = savefolder / (imfile.stem + '.json')
            with open(str(savefile), 'w') as f:
                json.dump(temp, f)
