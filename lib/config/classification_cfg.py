from ..helpers.config import config

classification_engine_model_path = config.get('CLASSIFICATION_ENGINE_MODEL_PATH', cast=str, default=None)