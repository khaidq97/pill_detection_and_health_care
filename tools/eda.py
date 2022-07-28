import os

import cv2
import shutil
from pathlib import Path 
from tqdm import tqdm

from lib.utils import utils

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


def pill_to_name_id(datasetFolder, trainTestFile, saveFolder):
    datasetFolder, saveFolder = Path(datasetFolder),  Path(saveFolder)
    if saveFolder.exists(): shutil.rmtree(str(saveFolder))
    os.makedirs(str(saveFolder))
    os.makedirs(str(saveFolder / 'Error'))

    trainTestInfo = list(open(str(trainTestFile), 'r'))

    for info in tqdm(trainTestInfo):
        type, name = info.strip().split()
        imgfile = datasetFolder / 'pill' / 'image' / name
        lbfile = datasetFolder / 'pill' / 'label' / (name[:-4] + '.json')
        presfile = datasetFolder / 'prescription' / 'label' /'VAIPE_P_TRAIN_{}.json'.format(name.split('_')[2])

        img = cv2.imread(str(imgfile))
        for data in utils.read_json_file(lbfile):
            try:
                bbox = utils.xywh2xyxy([data['x'], data['y'], data['w'], data['h']])
                pill_img = utils.crop_image(img, bbox)
            except:
                print('Error: ', str(imgfile))
                shutil.copy2(str(imgfile), str(saveFolder / 'Error' / imgfile.name))
                shutil.copy2(str(lbfile), str(saveFolder / 'Error' / lbfile.name))
                continue

            id = str(data['label'])
            for d in utils.read_json_file(str(presfile)):
        
                if d['label'] == 'drugname':
                    if id == str(d['mapping']):
                        saveFolder_ = saveFolder / type / id 
                        if saveFolder_.exists():
                            stts = [int(f.stem.split('_')[-1]) for f in saveFolder_.glob('*')]
                            stt = max(stts) if len(stts) else -1
                        else:
                            os.makedirs(str(saveFolder_))
                            stt = -1
                        namefile = "{}_{}_{}.jpg".format(id, d['text'], stt+1)
                        savefile = saveFolder_ / namefile
                        cv2.imwrite(str(savefile), pill_img)




if __name__ == '__main__':
    # eda_pill_id(folder='document/dataset/public_train/pill',
    #             savefolder='document/output/pill_id')

    pill_to_name_id(datasetFolder='document/dataset/public_train',
                    trainTestFile='document/dataset/train_val.txt',
                    saveFolder='document/output/pill_id_name')


