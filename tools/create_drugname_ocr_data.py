import os 
import shutil
import cv2 
from tqdm import tqdm
from pathlib import Path 
import random

from lib.utils import utils
random.seed(24)

def create_drugname_data_vietocr_format(folder, savefolder):
    folder, savefolder = Path(folder), Path(savefolder)
    if savefolder.exists(): shutil.rmtree(str(savefolder))
    os.makedirs(str(savefolder))

    logfile = open(str(savefolder/"log.txt"), 'w')
    parent_name = str(savefolder).split('/')[-1]
    imfiles = [x for x in folder.rglob('*') if x.suffix in ('.jpg', '.png')]
    i = 0
    for imfile in tqdm(imfiles):
        lbfile = str(imfile.parent / (imfile.stem + '.json')).replace('/image/', '/label/')
        im = cv2.imread(str(imfile))
        for data in utils.read_json_file(lbfile):
            if data['label'] == 'drugname':
                pres_img = utils.crop_image(im.copy(), data['box'])
                idx = data['mapping']
                text = data['text']
                name = "{}_{}_{}.png".format(text, idx, i)
                i += 1
                outfile = savefolder / name
                cv2.imwrite(str(outfile), pres_img)
                logfile.write("{}\t{}\n".format(parent_name + '/' + name, text))

def create_all_data_vietocr_format(folder, savefolder):
    folder, savefolder = Path(folder), Path(savefolder)
    if savefolder.exists(): shutil.rmtree(str(savefolder))
    os.makedirs(str(savefolder))

    logfile = open(str(savefolder/"log.txt"), 'w')
    parent_name = str(savefolder).split('/')[-1]
    imfiles = [x for x in folder.rglob('*') if x.suffix in ('.jpg', '.png')]
    i = 0
    for imfile in tqdm(imfiles):
        lbfile = str(imfile.parent / (imfile.stem + '.json')).replace('/image/', '/label/')
        im = cv2.imread(str(imfile))
        for data in utils.read_json_file(lbfile):
            pres_img = utils.crop_image(im.copy(), data['box'])
            text = data['text']
            label = data['label']
            if label == 'drugname':
                idx = data['mapping']
            else:
                idx = 'None'
            name = "{}_{}_{}_{}_{}.png".format(imfile.stem, label, idx, text, i)
            i += 1
            outfile = savefolder / name
            cv2.imwrite(str(outfile), pres_img)
            logfile.write("{}\t{}\n".format(parent_name + '/' + name, text))
    logfile.close()
    
    with open(str(savefolder/"log.txt"), 'r') as f:
        dataset = f.readlines()
    random.shuffle(dataset)
    n_train = int(0.8*len(dataset))
    trainfile = open(str(savefolder/"train.txt"), 'w')
    for line in dataset[:n_train]:
        trainfile.write(line)
    trainfile.close()

    testfile = open(str(savefolder/"test.txt"), 'w')
    for line in dataset[n_train:]:
        testfile.write(line)
    testfile.close()


if __name__ == '__main__':
    # create_drugname_data_vietocr_format(folder='document/dataset/public_train/prescription/image',
    #                                     savefolder='document/output/pres_vietocr')

    create_all_data_vietocr_format(folder='document/dataset/public_train/prescription/image',
                                        savefolder='document/output/pres_vietocr_all')
