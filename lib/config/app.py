from ..helpers.config import config

device = config.get('DEVICE', default=None, cast=str)
