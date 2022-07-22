# Check bbox detector
import cv2 
from pathlib import Path 
from lib.detection_engine import detection_engine
from lib.utils import utils

detectionEngine = detection_engine.DetectionEngine()

SHOW_SIZE = (1000,1000)
img_files = [p for p in Path('document/dataset/public_test/pill/image').rglob('*') if p.suffix in ('.jpg', '.png')]

i = 0
while True:
    imgfile = img_files[i]
    img = cv2.imread(str(imgfile))
    bboxes, scores = detectionEngine.predict(img)
    if bboxes is not None:
        texts = [str(round(score,2)) for score in scores]
        img = utils.draw_bboxes(img, bboxes, texts)
    
    cv2.imshow("PILL", cv2.resize(img, SHOW_SIZE))
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    elif k == ord('a') and i > 0:
        i -= 1
    elif k == ord('d') and i < len(img_files)-1:
        i += 1
cv2.destroyAllWindows()