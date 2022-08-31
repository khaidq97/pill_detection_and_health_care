import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
import sys 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from lib.controller import Controller


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='The path to dataset dir', type=str,
                        default=None)
parser.add_argument('--save_dir', help='The path to save results', type=str,
                        default=None)
cfg = parser.parse_args()


controller = Controller()

root = Path(cfg.root)
pill_dir = root / 'pill' / 'image'
pres_dir = root / 'prescription' / 'image'

controller.run_state(pill_dir=pill_dir,
                    pres_dir=pres_dir,
                    save_dir=cfg.save_dir)
