# Check bbox detector
import shutil
import cv2 
import os
from pathlib import Path 
from lib.utils import utils
#=================================Setting==============================================#SHOW_SIZE = (1000,1000)
COLOR = (100,127,255)

#======================================================================================#
def test_detection_engine(folder, savefolder):
    SHOW_SIZE = (1000,1000)
    from lib.detection_engine import detection_engine
    detectionEngine = detection_engine.DetectionEngine()

    folder, savefolder = Path(folder), Path(savefolder)
    if not savefolder.exists(): 
        os.makedirs(str(savefolder))
        os.makedirs(str(savefolder / 'debug'))
        with open(str(savefolder / 'id.txt'), 'w') as f:
            f.write(str(0))

    with open(str(savefolder / 'id.txt'), 'r') as f:
        i = int(f.readline())
    img_files = [p for p in Path(folder).rglob('*') if p.suffix in ('.jpg', '.png')]
    while True:
        imgfile = img_files[i]
        img = cv2.imread(str(imgfile))
        img_ = img.copy()
        bboxes, labels, scores = detectionEngine.predict(img)
        if bboxes is not None:
            texts = []
            for score, label in zip(scores, labels):
                text = "{}:{:.2f}".format(label, score)
                texts.append(text)
            img_ = utils.draw_bboxes(img_, bboxes, texts)

        #=================================================================================#
        save_names = [f.name for f in savefolder.glob('*') if f.suffix in ('.jpg', '.png')]
        status = 'saved' if imgfile.name in save_names else 'not save'

        img_ = cv2.resize(img_, SHOW_SIZE)
        x0, y0 = 20, 20
        cv2.putText(img_, f"{i}|{len(img_files)}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"{imgfile.name}", (10*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"Status: {status}", (30*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.imshow("PILL", img_)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(i))
            break
        elif k == ord('a') and i > 0:
            i -= 1
        elif k == ord('d') and i < len(img_files)-1:
            i += 1
        elif k == ord('s'):
            cv2.imwrite(str(savefolder / imgfile.name), img)
            cv2.imwrite(str(savefolder / 'debug' / imgfile.name), img_)
        elif k == ord('r'):
            shutil.rmtree(str(savefolder))
            os.makedirs(str(savefolder))
            os.makedirs(str(savefolder / 'debug'))
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(0))
            i = 0
        elif k == ord('e'):
            try:
                os.remove(str(savefolder / imgfile.name))
                os.remove(str(savefolder / 'debug' / imgfile.name))
            except: continue
    cv2.destroyAllWindows()

#=======================================================================================================================#
def test_pres_detector(folder, savefolder):
    SHOW_SIZE = (1000,1500)
    from lib.pres_ocr_engine.pres_detector import pres_detector
    from lib.config import app
    from lib.config import pres_ocr_cfg
    # print(pres_ocr_config.pres_yolov5_model_path)
    presDetector = pres_detector.PresDetector(app_config=app, pres_ocr_config=pres_ocr_cfg)

    folder, savefolder = Path(folder), Path(savefolder)
    if not savefolder.exists(): 
        os.makedirs(str(savefolder))
        os.makedirs(str(savefolder / 'debug'))
        with open(str(savefolder / 'id.txt'), 'w') as f:
            f.write(str(0))

    with open(str(savefolder / 'id.txt'), 'r') as f:
        i = int(f.readline())
    img_files = [p for p in Path(folder).rglob('*') if p.suffix in ('.jpg', '.png')]
    while True:
        imgfile = img_files[i]
        img = cv2.imread(str(imgfile))
        img_ = img.copy()
        bboxes, labels, scores = presDetector.predict(img)
        if bboxes is not None:
            texts = []
            for score, label in zip(scores, labels):
                text = "{}:{:.2f}".format(label, score)
                texts.append(text)
            img_ = utils.draw_bboxes(img_, bboxes, texts,
                                bbox_thickness=2,
                                txt_thickness=2,
                                txt_size=1)

        save_names = [f.name for f in savefolder.glob('*') if f.suffix in ('.jpg', '.png')]
        status = 'saved' if imgfile.name in save_names else 'not save'

        img_ = cv2.resize(img_, SHOW_SIZE)
        x0, y0 = 20, 20
        cv2.putText(img_, f"{i}|{len(img_files)}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"{imgfile.name}", (10*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(img_, f"Status: {status}", (30*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.imshow("PILL", img_)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(i))
            break
        elif k == ord('a') and i > 0:
            i -= 1
        elif k == ord('d') and i < len(img_files)-1:
            i += 1
        elif k == ord('s'):
            cv2.imwrite(str(savefolder / imgfile.name), img)
            cv2.imwrite(str(savefolder / 'debug' / imgfile.name), img_)
        elif k == ord('r'):
            shutil.rmtree(str(savefolder))
            os.makedirs(str(savefolder))
            os.makedirs(str(savefolder / 'debug'))
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(0))
            i = 0
        elif k == ord('e'):
            try:
                os.remove(str(savefolder / imgfile.name))
                os.remove(str(savefolder / 'debug' / imgfile.name))
            except: continue
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # test_detection_engine(folder='document/data/pill/val',
    #                         savefolder='document/output/test_detection_engine_log')

    test_pres_detector(folder='document/dataset/public_test/prescription/image',
                        savefolder='document/output/test_detection_engine_log')