import os
import cv2
import shutil
from pathlib import Path 
from tqdm import tqdm

from utils import utils

def eda_pill_id(folder, savefolder):
    folder, savefolder = Path(folder), Path(savefolder)
    
    if savefolder.exists(): shutil.rmtree(str(savefolder))
    os.makedirs(str(savefolder))

    imgfiles = [p for p in folder.rglob('*') if p.suffix in ('.jpg', '.png')]

    for imgfile in tqdm(imgfiles):
        lbfile = str(imgfile.parent / (imgfile.stem + '.json')).replace('/image/', '/label/')
        img = cv2.imread(str(imgfile))
        
        for data in utils.read_json_file(lbfile):
            bbox = utils.xywh2xyxy([data['x'], data['y'], data['w'], data['h']])
            id = str(data['label'])
            idfolder = savefolder / id
            if idfolder.exists():
                max_id = max([int(p.stem.split('_')[-1]) for p in idfolder.glob('*.jpg')])
                savefile = idfolder / "{}_{}.jpg".format(id, max_id+1)
            else:
                os.makedirs(str(idfolder))
                savefile = idfolder / "{}_{}.jpg".format(id, 0)

            pill_img = utils.crop_image(img, bbox)
            cv2.imwrite(str(savefile), pill_img)



if __name__ == '__main__':
    eda_pill_id(folder='dataset/public_train/pill',
                savefolder='out/pill_eda')


