import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def split_dataset(datafolder, savefolder, ratio=0.8):
    datafolder, savefolder = Path(datafolder), Path(savefolder)
    if savefolder.exists(): shutil.rmtree(str(savefolder))
    os.makedirs(str(savefolder))

    dataset = defaultdict(list)
    for imgfile in datafolder.rglob('*'):
        if imgfile.suffix not in ('.jpg', '.png'): continue
        name = imgfile.name
        pres = '_'.join(name.split('_')[:-1])
        dataset[pres].append(imgfile)
    
    image_train_folder = savefolder / 'image/train'
    image_val_folder = savefolder / 'image/val'
    label_train_folder = savefolder / 'label/train'
    label_val_folder = savefolder / 'label/val'
    os.makedirs(str(image_train_folder))
    os.makedirs(str(image_val_folder))
    os.makedirs(str(label_train_folder))
    os.makedirs(str(label_val_folder))
    log_text = open(str(savefolder/'train_val.txt'), 'w')
    for key, imgfiles in tqdm(dataset.items()):
        random.seed(24)
        random.shuffle(imgfiles)
        n_train = int(ratio * len(imgfiles))

        # Train
        for imgfile in imgfiles[:n_train]:
            imgdest = image_train_folder / imgfile.name
            lbfile = imgfile.parents[1] / 'label' / (imgfile.stem + '.json')
            lbdest = label_train_folder / lbfile.name

            shutil.copy2(str(imgfile), str(imgdest))
            shutil.copy2(str(lbfile), str(lbdest))
            log_text.write(f"train {imgfile.name}\n")

        # Validation
        for imgfile in imgfiles[n_train:]:
            imgdest = image_val_folder / imgfile.name
            lbfile = imgfile.parents[1] / 'label' / (imgfile.stem + '.json')
            lbdest = label_val_folder / lbfile.name

            shutil.copy2(str(imgfile), str(imgdest))
            shutil.copy2(str(lbfile), str(lbdest))
            log_text.write(f"val {imgfile.name}\n")


if __name__ == '__main__':
    split_dataset(datafolder='/home/khai/Desktop/VAIPE/dataset/public_train/pill/image',
                 savefolder='/home/khai/Desktop/VAIPE/data/pill',
                 ratio=0.85)