# Check bbox detector
import cv2 
import os
from pathlib import Path 
from lib.detection_engine import detection_engine
from lib.utils import utils

detectionEngine = detection_engine.DetectionEngine()
#=================================Setting==============================================#
SHOW_SIZE = (1000,1000)
FOLDER = Path('document/dataset/public_test/pill/image')
SAVEFOLDER = Path('document/fall_detection')
COLOR = (100,127,255)

#======================================================================================#
img_files = [p for p in Path(FOLDER).rglob('*') if p.suffix in ('.jpg', '.png')]

if not SAVEFOLDER.exists(): 
    os.makedirs(str(SAVEFOLDER))
    os.makedirs(str(SAVEFOLDER / 'debug'))

i = 0
while True:
    imgfile = img_files[i]
    img = cv2.imread(str(imgfile))
    img_ = img.copy()
    bboxes, scores = detectionEngine.predict(img)
    if bboxes is not None:
        texts = [str(round(score,2)) for score in scores]
        img_ = utils.draw_bboxes(img_, bboxes, texts)

    #===============
    save_names = [f.name for f in SAVEFOLDER.glob('*') if f.suffix in ('.jpg', '.png')]
    status = 'saved' if imgfile.name in save_names else 'not save'

    img_ = cv2.resize(img_, SHOW_SIZE)
    x0, y0 = 20, 20
    cv2.putText(img_, f"{i}|{len(img_files)}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
    cv2.putText(img_, f"{imgfile.name}", (10*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
    cv2.putText(img_, f"Status: {status}", (30*x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2, cv2.LINE_AA)
    cv2.imshow("PILL", img_)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    elif k == ord('a') and i > 0:
        i -= 1
    elif k == ord('d') and i < len(img_files)-1:
        i += 1
    elif k == ord('s'):
        cv2.imwrite(str(SAVEFOLDER / imgfile.name), img)
        cv2.imwrite(str(SAVEFOLDER / 'debug' / imgfile.name), img_)
cv2.destroyAllWindows()