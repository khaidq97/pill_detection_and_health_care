import os
import shutil
import cv2
import json 
from pathlib import Path 

from lib.utils import utils
from lib.controller import controller

controller = controller.Controller()

COLOR = (100,127,255)

def test_controller(pillfolder, presfolder, savefolder, reset=False):
    pillfolder, presfolder, savefolder = Path(pillfolder), Path(presfolder) ,Path(savefolder)

    if reset:
        shutil.rmtree(str(savefolder))
    if not savefolder.exists(): 
        os.makedirs(str(savefolder))
        os.makedirs(str(savefolder / 'debug'))
        with open(str(savefolder / 'id.txt'), 'w') as f:
            f.write(str(0))
    
    pill2restjson = savefolder / 'pill2res.json'
    if pill2restjson.exists():
        with open(str(pill2restjson)) as f:
            pill2res = json.load(f)
    else:
        print('Creating pill map pres')
        pill2res = utils.pills_to_pres_name(pillfolder, presfolder)
        with open(str(pill2restjson), 'w') as f:
            json.dump(pill2res, f)

    with open(str(savefolder / 'id.txt'), 'r') as f:
        i = int(f.readline())
    pillfiles = [p for p in Path(pillfolder).rglob('*') if p.suffix in ('.jpg', '.png')]

    print("Running...")
    while True:
        pillfile = pillfiles[i]
        presfile = presfolder / pill2res[pillfile.name]

        pill_im = cv2.imread(str(pillfile))
        pres_im = cv2.imread(str(presfile))
        img_ = pill_im.copy()
        gt_im = pill_im.copy()

        bboxes, ids, scores = controller.predict(pill_im, pres_im)

        if bboxes is not None:
            texts = []
            for id, score in zip(ids, scores):
                text = "{}:{:.2f}".format(id, score)
                texts.append(text)
            img_ = utils.draw_bboxes(img_, bboxes, texts,
                                bbox_thickness=5,
                                txt_thickness=7,
                                txt_size=3)
        
        lbfile = pillfile.parents[1] / 'label' / (pillfile.stem + '.json')
        if lbfile.exists():
            gt_im = utils.draw_pill_groundtruh(gt_im, lbfile)
        
        save_names = [f.name for f in savefolder.glob('*') if f.suffix in ('.jpg', '.png')]
        status = 'saved' if pillfile.name in save_names else 'not save'

        img_ = cv2.resize(img_, (1000,1000))
        x0, y0 = 20, 20
        cv2.putText(img_, f"{i}|{len(pillfiles)}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"{pillfile.name}", (10*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"Status: {status}", (30*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.imshow("PILL", img_)
        if lbfile.exists():
            cv2.imshow("PILL GT", cv2.resize(gt_im, (1000,1000)))
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(i))
            break
        elif k == ord('a') and i > 0:
            i -= 1
        elif k == ord('d') and i < len(pillfiles)-1:
            i += 1
        elif k == ord('s'):
            cv2.imwrite(str(savefolder / pillfile.name), pill_im)
            cv2.imwrite(str(savefolder / 'debug' / pillfile.name), img_)
        elif k == ord('r'):
            shutil.rmtree(str(savefolder))
            os.makedirs(str(savefolder))
            os.makedirs(str(savefolder / 'debug'))
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(0))
            i = 0
        elif k == ord('e'):
            try:
                os.remove(str(savefolder / pillfile.name))
                os.remove(str(savefolder / 'debug' / pillfile.name))
            except: continue
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_controller(pillfolder='document/dataset/public_test/pill/image',
                    presfolder='document/dataset/public_test/prescription/image',
                    savefolder='document/output/test_controller_log',
                    reset=True)

    # test_controller(pillfolder='document/dataset/public_train/pill/image',
    #                 presfolder='document/dataset/public_train/prescription/image',
    #                 savefolder='document/output/test_controller_log',
    #                 reset=True)


