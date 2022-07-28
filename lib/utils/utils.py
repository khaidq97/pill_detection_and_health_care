import json
import random
import cv2
import numpy as np
from pathlib import Path 

def read_json_file(jsonfile):
    with open(str(jsonfile)) as f:
        data = json.load(f)
    return data

def xywh2xyxy(bbox):
    '''
    COCO format to voc format
    '''
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def yolo2xywh(bbox, w, h):
    """
    yolo to coco format
    """
    xmid, ymid = bbox[0]*w, bbox[1]*h 
    w_, h_ = bbox[2]*w, bbox[3]*h 
    xmin = xmid - w_ / 2
    ymin = ymid - h_ / 2
    bbox[0] = xmin
    bbox[1] = ymin 
    bbox[2] = w_ 
    bbox[3] = h_ 
    return bbox

def xyxy2xywh(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    w = xmax - xmin 
    h = ymax - ymin
    bbox[2] = w
    bbox[3] = h
    return bbox


def crop_image(image, box):
    h, w = image.shape[:2]
    if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
        raise ValueError("Cropping Box is not valid")
    if box[0] > w or box[1] > h or box[2] > w or box[3] > h:
        raise ValueError("Cropping Box is not valid")
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("Cropping Box is not valid")
    return image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

def draw_bboxes(img, bboxes, texts=None, 
                    bbox_thickness=5,
                    txt_thickness=3,
                    txt_size=1,
                    bbox_color=(114,104,72),
                    txt_color=(26,87,247)):
    if not isinstance(type(bboxes), np.ndarray):
        bboxes = np.array(bboxes)
    bboxes = bboxes.astype(int)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cv2.rectangle(img, bbox[:2], bbox[2:], bbox_color, bbox_thickness)
        if texts is not None:
            x0 = bbox[0]
            y0 = bbox[1]
            text = str(texts[i])
            cv2.putText(img, text, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, txt_thickness)
    return img

def show_pill_label(imgfile, lbfile, size=(1000,1000)):
    img = cv2.imread(str(imgfile))
    bboxes, labels = [], []
    for data in read_json_file(lbfile):
        bbox = [data['x'], data['y'], data['w'], data['h']]
        label = data['label']
        bbox = xywh2xyxy(bbox)
        bboxes += [bbox]
        labels += [label]
    
    img = draw_bboxes(img, bboxes, labels)
    cv2.imshow("PILL Image", cv2.resize(img, size))
    cv2.waitKey()
    cv2.destroyAllWindows()


def pills_to_pres_name(pillfolder, presfolder):
    pillnames = [x.name for x in Path(pillfolder).glob('*') if x.suffix in ('.jpg', '.png')]
    presnames = [x.name for x in Path(presfolder).glob('*') if x.suffix in ('.jpg', '.png')]
    result = {}
    for pillname in pillnames:
        pill_id = pillname.split('_')[2]
        for presname in presnames:
            pres_id = presname.split('_')[-1][:-4]
            if pill_id == pres_id:
                result[pillname] = presname 
                break 
    return result


def draw_pill_groundtruh(im, lbfile):
    bboxes, texts = [], []
    for data in read_json_file(str(lbfile)):
        bbox = [data['x'], data['y'], data['w'], data['h']]
        label = data['label']
        bbox = xywh2xyxy(bbox)
        bboxes += [bbox]
        texts += [label]

    im = draw_bboxes(im, bboxes, texts,
                    bbox_thickness=5,
                    txt_thickness=7,
                    txt_size=3)
    return im
