#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import os
import shutil 
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd 
from lib.utils import utils

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#====================================================================================#
class COCOeval_wmAP(COCOeval):
    def __init__(self, gt_coco, res_coco, iouType='segm', num_cls=108, alpha=10):
        super(COCOeval_wmAP, self).__init__(gt_coco, res_coco, iouType)
        self.num_cls = num_cls
        self.weights = np.array([1 if i!=num_cls-1 else alpha for i in range(num_cls)])

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                s[s==-1] =0
                mean_s = np.mean(np.average(s, weights=self.weights, axis=-2))
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

#=====================================================================================#
def create_grountruth_yolo_to_coco(folder, savefile):
    folder, savefile = Path(folder), Path(savefile)
    if not savefile.parent.exists(): os.makedirs(str(savefile.parent))
    imfiles = [x for x in folder.rglob('*') if x.suffix in ('.jpg', '.png')]

    names, annotations, images = [], [], []
    id = 0
    for imfile in tqdm(imfiles):
        lbfile = str(imfile.parent / (imfile.stem + '.txt')).replace('/images/', '/labels/')
        h, w = cv2.imread(str(imfile)).shape[:2]
        
        anno = False
        with open(lbfile, 'r') as file:
            for line in file:
                data = line.strip().split()
                name = int(data[0])
                bbox = utils.yolo2xywh([float(d) for d in data[1:]], w, h)
                names.append(name)
                annotations.append({
                    'area':bbox[2]*bbox[3],
                    'bbox': bbox,
                    'category_id': int(name),
                    'id': id,
                    'image_id': imfile.stem,
                    'iscrowd': 0,
                    'segmentation': []
                })
                id += 1
                anno = True
        if anno:
            images.append({
                'file_name': imfile.name,
                'id': imfile.stem,
                'width': w,
                'height': h
            })
    # create categories
    categories = []
    for i in range(108):
        categories.append({
            'id': i,
            'name': str(i),
            'supercategory': ''
        })
    
    dataset = {
        'categories': categories,
        'annotations': annotations,
        'images': images
    }
    with open(str(savefile), 'w') as f:
        json.dump(dataset, f)

#==================================================================================================#
def create_prediction_coco(csvfile, jsonfile):
    csvfile, jsonfile = Path(csvfile), Path(jsonfile)
    if not jsonfile.parent.exists(): os.makedirs(str(jsonfile.parent))

    dataset = []
    df = pd.read_csv(str(csvfile))
    for i in tqdm(range(len(df))):
        image_name = df['image_name'].loc[i]
        class_id = int(df['class_id'].loc[i])
        confidence_score = float(df['confidence_score'].loc[i])
        x_min, y_min, x_max, y_max = float(df['x_min'].loc[i]), float(df['y_min'].loc[i]), float(df['x_max'].loc[i]), float(df['y_max'].loc[i])
        bbox = utils.xyxy2xywh([x_min, y_min, x_max, y_max])
        dataset.append({
            'image_id': image_name[:-4],
            'category_id': class_id,
            'bbox': bbox,
            'score': confidence_score
        })
    jsonstr = json.dumps(dataset)
    with open(str(jsonfile), 'w') as f:
        f.write(jsonstr)

#===================================================================================================#
def evalue(gtjsonfile, predjsonfile):
    anno = COCO(str(gtjsonfile))
    pred = anno.loadRes(str(predjsonfile))
    cocoEval = COCOeval_wmAP(anno, pred, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    return map, map50

#================================================================================================#
def run_from_resultcsv_file(outfolder, infolder='', result_csvfile='', reset=False):
    outfolder = Path(outfolder)
    if reset:
        shutil.rmtree(str(outfolder))

    if not outfolder.exists(): os.makedirs(str(outfolder))
    gtjsonfile = outfolder / 'gt_results.json'

    if not gtjsonfile.exists():
        print("Creating: grountruth jsonfile", str(gtjsonfile))
        create_grountruth_yolo_to_coco(folder=infolder, savefile=gtjsonfile)

    if not os.path.isfile(str(outfolder / 'pred_results.csv')) and not  os.path.isfile(result_csvfile):
        from tools.create_result import result_of_detection_engine
        result_csvfile = outfolder / 'pred_results.csv'
        print("Creating predict csvfile:", result_csvfile)
        result_of_detection_engine(infolder, result_csvfile)
    
    predjsonfile = outfolder / 'pred_results.json'
    if not predjsonfile.exists():
        print("Creating predict jsonfile", str(predjsonfile))
        create_prediction_coco(csvfile=result_csvfile, jsonfile=predjsonfile)
    
    print("\n\nBENCHMARK:")
    evalue(gtjsonfile, predjsonfile)




if __name__ == '__main__':
    run_from_resultcsv_file(infolder='document/output/pill_yolo_name_minival/images',
                            outfolder='document/benchmark/minival',
                            result_csvfile='',
                            reset=False)
    