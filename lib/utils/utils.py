import json
import random
import cv2

def read_json_file(jsonfile):
    with open(str(jsonfile)) as f:
        data = json.load(f)
    return data

def xywh2xyxy(bbox):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def crop_image(image, box):
    return image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

def draw_bboxes(img, bboxes, texts=None):
    color = [random.randint(0,255) for _ in range(3)]
    colors = [color for _ in range(len(bboxes))]

    for i in range(len(bboxes)):
        bbox, color = bboxes[i], colors[i]
        cv2.rectangle(img, bbox[:2], bbox[2:], color, 10)
        if texts is not None:
            x0 = bbox[0]
            y0 = bbox[1]
            text = str(texts[i])
            cv2.putText(img, text, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)
    return img