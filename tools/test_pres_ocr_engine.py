import os 
import cv2
from pathlib import Path 
from lib.utils import utils
COLOR = (100,127,255)

def test_pres_ocr_engine(folder, savefolder, size=(1000,1300)):
    from lib.pres_ocr_engine import pres_ocr_engine
    presOCREngine = pres_ocr_engine.PresOCREngine()

    folder, savefolder = Path(folder), Path(savefolder)
    folder, savefolder = Path(folder), Path(savefolder)
    if not savefolder.exists(): 
        os.makedirs(str(savefolder))
        os.makedirs(str(savefolder / 'debug'))
        with open(str(savefolder / 'id.txt'), 'w') as f:
            f.write(str(0))

    with open(str(savefolder / 'id.txt'), 'r') as f:
        i = int(f.readline())
    imfiles = [p for p in Path(folder).rglob('*') if p.suffix in ('.jpg', '.png')]
    while True:
        imfile = imfiles[i]
        img = cv2.imread(str(imfile))
        im = img.copy()
        result =  presOCREngine.predict(im)
        if result is not None:
            bboxes, texts = [], []
            for data in result:
                bbox, label, text, ids = data['bbox'], data['label'], data['text'], data['ids']
                text += f"-id:{ids}"
                bboxes.append(bbox)
                texts.append(text)
                print("{} => {}".format(label, text))
            im = utils.draw_bboxes(im, bboxes, texts,
                                    bbox_thickness=2,
                                    txt_thickness=2,
                                    txt_size=0.7)

        save_names = [f.name for f in savefolder.glob('*') if f.suffix in ('.jpg', '.png')]
        status = 'saved' if imfile.name in save_names else 'not save'

        im = cv2.resize(im, size)
        x0, y0 = 20, 20
        cv2.putText(im, f"{i}|{len(imfiles)}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(im, f"{imfile.name}", (10*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.putText(im, f"Status: {status}", (30*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
        cv2.imshow("PRES", im)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(i))
            break
        elif k == ord('a') and i > 0:
            i -= 1
        elif k == ord('d') and i < len(imfiles)-1:
            i += 1
        elif k == ord('s'):
            cv2.imwrite(str(savefolder / imfile.name), img)
            cv2.imwrite(str(savefolder / 'debug' / imfile.name), im)
        elif k == ord('r'):
            shutil.rmtree(str(savefolder))
            os.makedirs(str(savefolder))
            os.makedirs(str(savefolder / 'debug'))
            with open(str(savefolder / 'id.txt'), 'w') as f:
                f.write(str(0))
            i = 0
        elif k == ord('e'):
            try:
                os.remove(str(savefolder / imfile.name))
                os.remove(str(savefolder / 'debug' / imfile.name))
            except: continue
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_pres_ocr_engine(folder='document/dataset/public_test/prescription/image',
                            savefolder='document/output/pres_ocr_temp_test')
