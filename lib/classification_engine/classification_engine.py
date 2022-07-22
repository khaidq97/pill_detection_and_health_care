from ..config import app as app_config
from ..config import classification_cfg as classification_config

class ClassificationEngine(object):
    def __init__(self, app_config=app_config, classification_config=classification_config):
        classification_model = classification_config.classification_model

    def predict(self):
        pass