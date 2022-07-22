from ..helpers.config import config

DEVICES = {
    'CPU': 'cpu',
    'GPU': 'cuda'
}
device = config.get('DEVICE', default=DEVICES['CPU'], cast=str)
