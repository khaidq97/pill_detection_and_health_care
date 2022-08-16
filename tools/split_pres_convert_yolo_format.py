import json
import os 
import shutil
import random
from PIL import Image, ExifTags
from tqdm import tqdm
from pathlib import Path


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
        dataset = json.load(f)
    file = open(str(labeldest), 'w')
    for data in dataset:
        label = data['label']
        if label in ('drugname', 'diagnose'):
            lb = 0 if label == 'drugname' else 1
            box = data['box']
            x_mid = (box[0] + box[2]) / (2*w)
            y_mid = (box[1] + box[3]) / (2*h)
            w_norm = (box[2] - box[0]) / w 
            h_norm = (box[3] - box[1]) / h
            file.write(f'{lb} {x_mid} {y_mid} {w_norm} {h_norm}\n')
            # file.write("{} {} {} {} {}\n".format(lb, x_mid, y_mid, w_norm, h_norm))
    file.close()


#===============================================================================================#
DATA_FOLDER = Path('document/dataset/public_train/prescription')
SAVE_FOLDER = Path('document/output/pres_yolo')
ratio = 0.9

random.seed(24)
#===============================================================================================#
if os.path.exists(str(SAVE_FOLDER)): shutil.rmtree(str(SAVE_FOLDER))
os.makedirs(str(SAVE_FOLDER))

IMAGE_TRAIN_FOLDER = SAVE_FOLDER / 'images' / 'train'
LABEL_TRAIN_FOLDER = SAVE_FOLDER / 'labels' / 'train'
IMAGE_VAL_FOLDER = SAVE_FOLDER / 'images' / 'val'
LABEL_VAL_FOLDER = SAVE_FOLDER / 'labels' / 'val'
os.makedirs(str(IMAGE_TRAIN_FOLDER))
os.makedirs(str(LABEL_TRAIN_FOLDER))
os.makedirs(str(IMAGE_VAL_FOLDER))
os.makedirs(str(LABEL_VAL_FOLDER))

imfiles = [x for x in DATA_FOLDER.rglob('*') if x.suffix in ('.jpg', '.png')]
random.shuffle(imfiles)

n_train = int(ratio * len(imfiles))
# Train
print('TRAIN:')
for imfile in tqdm(imfiles[:n_train]):
    imdest = IMAGE_TRAIN_FOLDER / imfile.name
    copy_img_label(imfile, imdest)

# val
print('VAL:')
for imfile in tqdm(imfiles[n_train:]):
    imdest = IMAGE_VAL_FOLDER / imfile.name
    copy_img_label(imfile, imdest)



