from ..helpers.config import config

model_name = config.get('MODEL NAME', default="vgg_seq2seq", cast=str)
