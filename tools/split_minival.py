import random
import os 
import shutil
from pathlib import Path 
from tqdm import tqdm


#=======================================================================#
FOLDER = Path('document/output/pill_yolo_name/images/val')
SAVE_FOLDER = Path('document/output/pill_yolo_name_minival')
n = 500
if SAVE_FOLDER.exists():shutil.rmtree(str(SAVE_FOLDER))
os.makedirs(str(SAVE_FOLDER))

IMAGES_FOLDER = SAVE_FOLDER / 'images'
LABELS_FOLDER = SAVE_FOLDER / 'labels'
IMAGES_FOLDER.mkdir(), LABELS_FOLDER.mkdir()
random.seed(24)

imfiles = [x for x in FOLDER.rglob('*') if x.suffix in ('.jpg', '.png')]
random.shuffle(imfiles)

for imfile in tqdm(imfiles[:n]):
    lbfile = str(imfile.parent / (imfile.stem + '.txt')).replace('/images/', '/labels/')
    imdest = IMAGES_FOLDER / imfile.name
    lbdest = LABELS_FOLDER / Path(lbfile).name

    shutil.copy2(str(imfile), str(imdest))
    shutil.copy2(str(lbfile), str(lbdest))



