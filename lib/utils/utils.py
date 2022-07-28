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

def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def is_in_line_boxes(box1, box2, thresh_y_diff_1, thresh_y_diff_2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    if abs(y21 - y11) > thresh_y_diff_1:
        return False
    if y22 - y12 > thresh_y_diff_2:
        return False
    
    return True

def find_in_line_boxes(box1, box_list, thresh_y_diff_1, thresh_y_diff_2):
    in_line_boxes = []
    in_line_indexes = []
    for i, box2 in enumerate(box_list):
        if box2 == box1:
            continue
        if is_in_line_boxes(box1, box2, thresh_y_diff_1, thresh_y_diff_2):
            in_line_boxes.append(box2)
            in_line_indexes.append(i)

    return in_line_boxes, in_line_indexes

