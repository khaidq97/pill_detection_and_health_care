import cv2
import os
from pathlib import Path 
from tqdm import tqdm 
import pandas as pd

from lib.utils import utils

def result_of_detection_engine(datafolder, savefile):
    from lib.detection_engine import detection_engine
    detectionEngine = detection_engine.DetectionEngine()
    if not Path(savefile).parent.exists(): os.makedirs(str(Path(savefile).parent))

    imfiles = [x for x in Path(datafolder).rglob('*') if x.suffix in ('.jpg', '.png')]
    
    df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    i = 0
    for imfile in tqdm(imfiles):
        im = cv2.imread(str(imfile))
        bboxes, labels, scores = detectionEngine.predict(im)
        if bboxes is not None:
            for bbox, lb, sc in zip(bboxes, labels, scores):
                df.loc[i] = [imfile.name, lb, sc, bbox[0], bbox[1], bbox[2], bbox[3]]
                i += 1
    print("Save at: ", savefile)
    df.to_csv(str(savefile), index=False)

def result_of_groundtruth(datafolder, savefile):
    if not Path(savefile).parent.exists(): os.makedirs(str(Path(savefile).parent))
    imfiles = [x for x in Path(datafolder).rglob('*') if x.suffix in ('.jpg', '.png')]
    df = pd.DataFrame(columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    i = 0
    for imfile in tqdm(imfiles):
        lbfile = imfile.parent / (imfile.stem + '.json')
        lbfile = str(lbfile).replace('/image/', '/label/')
        im = cv2.imread(str(imfile))
        for data in utils.read_json_file(lbfile):
            bbox = utils.xywh2xyxy([data['x'], data['y'], data['w'], data['h']])
            try:
                _ = utils.crop_image(im, bbox)
            except: continue
            text = "{} {} {} {} {} {}\n".format(imfile.name, data['label'], bbox[0], bbox[1], bbox[2], bbox[3])
            df.loc[i] = [imfile.name, data['label'], 1.0, bbox[0], bbox[1], bbox[2], bbox[3]]
            i += 1
    print("Save at: ", savefile)
    df.to_csv(str(savefile), index=False)



if __name__ == '__main__':
    # result_of_groundtruth(datafolder='document/data/pill/val',
    #                         savefile='document/results/gt_result.csv')
    
    # result_of_detection_engine(datafolder='document/data/pill/val/image',
    #                             savefile='document/results/detect_result.csv')

    result_of_detection_engine(datafolder='document/dataset/public_test/pill/image',
                                savefile='document/results/results.csv')
