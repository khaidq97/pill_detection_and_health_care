from ..helpers.config import config

craft_refine_model_path = config.get('CRAFT REFINE MODEL PATH', 
                                     default="trained_models/text_detection/craft/weights/craft_refiner_CTW1500.pth",
                                     cast=str)

craft_detection_model_path = config.get('CRAFT DETECTION MODEL PATH',
                                        default="trained_models/text_detection/craft/weights/craft_mlt_25k.pth",
                                        cast=str)