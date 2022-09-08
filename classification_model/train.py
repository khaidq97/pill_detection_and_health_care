import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2 
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.optim import Adam
import logging
from datetime import datetime
import sys
import shutil
from sklearn.utils import compute_class_weight




# from utils import preprocessing_dataframe, get_drugname_2_id, get_dotorname_2_id, cls_metrics, save_checkpoint, load_checkpoint
from model.model import DrugClassificationModel
from model_v2.model import DrugClassificationModel_v2
from model_v3.model import DrugClassificationModel_v3
from dataset import VAIPEDatsets
from validation_fn import validation_fn

from utils import *
from config import Config

data_path = "/media/case.kso@kaopiz.local/New Volume/hiennt/pill_detection"

def training_fn(train_loader, val_loader, model, epochs, optimizer, loss_fn, saved_dir, resume_train):
    logging.info("....................Training ...................")
    best_score = 0
    last_epoch = 1
    
    # resume training
    saved_dir_v2 =  saved_dir.split("/")[-1]+"_mapping.pt"
    saved_dir_v2 = os.path.join(saved_dir, saved_dir_v2)
    saved_dir = os.path.join(saved_dir, saved_dir.split("/")[-1]+".pt")
    if os.path.exists(saved_dir) and resume_train==True:
        last_epoch, train_f1, val_f1 = load_checkpoint(torch.load(saved_dir), model)
        logging.info("Loading checkpoint and continue tranining:{}".format(saved_dir))
        logging.info('Epoch: {0}: , Training F1_score: {1} , Val F1_score: {2}'.format(last_epoch, train_f1, val_f1))
        best_score = val_f1
        
    for epoch in range(last_epoch, epochs+1):
        model.train()
        logging.info('Epoch:{}'.format(epoch))
        loss_train_total = 0
        predictions, true_labels = [], []
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for idx, batch in enumerate(progress_bar):
            image_input = batch["image_input"].to(device)
            diagnose_input = batch["diagnose_input"].to(device)
            bboxes_input = batch["bbox_input"].to(device)
            doctor_input = batch["doctor_input"].to(device)
            quantity_input = batch["quantity_input"].to(device)
            drugnames_input = batch["drugnames_input"].to(device)
            targets = batch["targets"].to(device)
            

            preds = model(image_input, diagnose_input, drugnames_input, bboxes_input, doctor_input, quantity_input)

            loss = loss_fn(preds, targets)
            loss_train_total += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # softmax_fn = torch.nn.Softmax(dim=1)
            # preds = softmax_fn(preds)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            targets = torch.argmax(targets, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(targets)

        logging.info('training_loss: {:.3f}'.format(loss_train_total/int((idx+1))))
        train_acc, train_f1 = cls_metrics(true_labels, predictions)
        logging.info("\t Training accuracy score: {}".format(train_acc))
        logging.info("\t Training f1 score: {}".format(train_f1))
        
        logging.info("....................Validation ...................")
        val_acc, val_f1, val_loss = validation_fn(val_loader, model, loss_fn, epoch, device)
        logging.info("validation loss: {:.3f}".format(val_loss))
        logging.info("\t Validation accuracy score: {}".format(val_acc))
        logging.info("\t Validation f1 score: {}".format(val_f1))
        # if True:
        if val_f1 > best_score:
            best_score = val_f1
            checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_f1_score': train_f1, 'val_f1_score': val_f1}
            # saved_dir_v2_ = saved_dir_v2.replace(".pt", "_"+str(epoch)+".pt")
            # save_checkpoint(checkpoint, saved_dir_v2_)
            # saved_dir_ = saved_dir.replace(".pt", "_"+str(epoch)+".pt")
            save_checkpoint(checkpoint, saved_dir)
            logging.info("Saving new checkpoint at {}".format(saved_dir))
    
if __name__ == "__main__":
    backbone = Config.BACKBONE
    resume_train = Config.RESUME_TRAIN
    num_classes = Config.NUM_CLASSE
    train_name = Config.TRAIN_DATA
    bert_model = Config.BERT_MODEL
    use_diagnose = Config.USE_DIAGNOSE
    use_drugname = Config.USE_DRUGNAME
    use_additional = Config.USE_ADDITIONAL
    
    saved_dir = "saved_models/"+backbone+"_"+str(use_diagnose)+"_"+str(use_drugname)+"_"+str(use_additional)+"_tiny_v3_without_bbox" # v1 without 25 49 agumentation
    if os.path.exists(saved_dir) is False:
        os.mkdir(saved_dir)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(saved_dir, str(datetime.now().strftime("%d-%m-%Y:%H:%M:%S"))+".log"), 
                        format='%(asctime)s %(message)s', filemode='w',
                        )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
    
    dataframe = pd.read_csv(os.path.join("pill_csv_fix_bbox", train_name))
    # dataframe = dataframe.sample(5000)
    train_df, val_df, dataframe = preprocessing_dataframe(dataframe)
    test_df = pd.read_csv(os.path.join("pill_csv_v2", "public_test_gt_encoded.csv"))
    _, _, test_df = preprocessing_dataframe(test_df)
    
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=False, lower_case=True)
    
    # get drug name to id
    drugname_2_id = load_json(os.path.join("mapping","drugname_2_id.json"))
    doctor_2_id = load_json(os.path.join("mapping", "doctor_2_id.json"))
    
    train_folder = os.path.join(data_path, "public_train", "original_train+gen_case")
    test_folder = os.path.join(data_path, "public_test", "gt")
    
    train_dataset = VAIPEDatsets(train_folder, dataframe, tokenizer, num_classes, drugname_2_id, doctor_2_id)
    unlabeled_dataset = VAIPEDatsets(test_folder, test_df.head(int(0.5*len(test_df))), tokenizer, num_classes, drugname_2_id, doctor_2_id)
    val_dataset = VAIPEDatsets(test_folder, test_df, tokenizer, num_classes, drugname_2_id, doctor_2_id)

    logging.info("Length training dataset: {}".format(train_dataset.__len__()))
    logging.info("Length unlabeled dataset: {}".format(unlabeled_dataset.__len__()))
    logging.info("Length validaion dataset: {}".format(val_dataset.__len__()))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DrugClassificationModel(num_classes=num_classes, bert_model=bert_model, use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
    # model = DrugClassificationModel_v2(backbone=backbone, num_classes=num_classes, bert_model=bert_model, use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
    model = DrugClassificationModel_v3(backbone=backbone, num_classes=num_classes, bert_model=bert_model, use_diagnose=use_diagnose, use_drugname=use_drugname, use_additional=use_additional)
    model = model.to(device)
    
    class_weights = compute_class_weight(class_weight = "balanced",
                                    classes = np.unique(dataframe['id']),
                                    y = dataframe['id']                                                    
                                    )
    # class_weights[107] = class_weights[107] + 0.5
    class_weights = torch.from_numpy(class_weights).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) 
    # loss_fn = torch.nn.CrossEntropyLoss() 
    # loss_fn = torch.nn.NLLLoss()
    # loss_fn = torch.nn.KLDivLoss()
    
    # optimizer = Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-5)
    # optimizer = Adam(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    training_fn(train_loader, val_loader, model, epochs=Config.EPOCHS, optimizer=optimizer, loss_fn=loss_fn, saved_dir=saved_dir, resume_train=resume_train)