import json
from ..config import pres_ocr_cfg

with open(pres_ocr_cfg.drugname_to_ids, 'r') as f:
    drugname_to_ids = json.load(f)

def drugname_postprocess(text):
    if ')' in text:
        text = ' '.join(text.split()[1:])
    # print(text)
    ids = drugname_to_ids[text]
    return ids, text
