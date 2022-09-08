
class Config():
    TRAIN_DATA = "public_train_original_encoded+gen_case.csv"
    NUM_CLASSE = 107+1
    BERT_MODEL = "vinai/phobert-base"  
    BACKBONE = "swin"
    RESUME_TRAIN = False
    USE_DIAGNOSE = True
    USE_DRUGNAME = False
    USE_ADDITIONAL = False
    EPOCHS = 50
