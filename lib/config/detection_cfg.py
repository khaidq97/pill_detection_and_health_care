from ..helpers.config import config

detection_engine_model_path = config.get('DETECTION_ENGINE_MODEL_PATH', cast=str,
            default=None)