import json
import os 
import shutil
import random
from PIL import Image, ExifTags
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def copy_img_label(imgfile, imgdest):
    imgfile, imgdest = Path(imgfile), Path(imgdest)
    labelfile = imgfile.parents[1] / 'label' / (imgfile.stem + '.json')
    labeldest = str(imgdest.parent / (imgdest.stem + '.txt')).replace('/images/', '/labels/') 
    shutil.copy2(str(imgfile), str(imgdest))

    img = Image.open(str(imgfile))
    w,h = exif_size(img)
    with open(str(labelfile)) as f:
        data = json.load(f)
    file = open(str(labeldest), 'w')
    for box in data:
        label = box['label']
        x_mid = (box["x"]/w + box["x"]/w + box["w"]/w)/2
        y_mid = (box["y"]/h + box["y"]/h + box["h"]/h)/2
        w_norm = box["w"]/w
        h_norm = box["h"]/h
        file.write(f'{label} {x_mid} {y_mid} {w_norm} {h_norm}\n')
    file.close()



#=================================================================================#
DATA_FOLDER = Path('/home/khai/Desktop/VAIPE/dataset/public_train/pill/image')
SAVE_FOLDER = Path('/home/khai/Desktop/VAIPE/ai4vn-vaipe-baseline/data/ai4vn')

#================================================================================#
if os.path.exists(str(SAVE_FOLDER)): shutil.rmtree(str(SAVE_FOLDER))
os.makedirs(str(SAVE_FOLDER))

dataset = defaultdict(list)
for imgfile in DATA_FOLDER.rglob('*'):
    if imgfile.suffix not in ('.jpg', '.png'): continue
    name = imgfile.name

    pres = '_'.join(name.split('_')[:-1])
    dataset[pres].append(imgfile)
# print(dataset)

IMAGE_TRAIN_FOLDER = SAVE_FOLDER / 'images' / 'train'
LABEL_TRAIN_FOLDER = SAVE_FOLDER / 'labels' / 'train'
IMAGE_VAL_FOLDER = SAVE_FOLDER / 'images' / 'val'
LABEL_VAL_FOLDER = SAVE_FOLDER / 'labels' / 'val'
os.makedirs(str(IMAGE_TRAIN_FOLDER))
os.makedirs(str(LABEL_TRAIN_FOLDER))
os.makedirs(str(IMAGE_VAL_FOLDER))
os.makedirs(str(LABEL_VAL_FOLDER))

ratio = 0.8
random.seed(24)
# Train
for key, imgfiles in tqdm(dataset.items()):
    random.shuffle(imgfiles)
    n_train = int(ratio * len(imgfiles))

    # Train
    for imgfile in imgfiles[:n_train]:
        imgdest = IMAGE_TRAIN_FOLDER / imgfile.name
        copy_img_label(imgfile, imgdest)
    
    # Val
    for imgfile in imgfiles[n_train:]:
        imgdest = IMAGE_VAL_FOLDER / imgfile.name
        copy_img_label(imgfile, imgdest)


