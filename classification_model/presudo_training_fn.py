import os
import numpy as np
import torch
import logging
import tqdm
from torchreid.utils.avgmeter import AverageMeter

from utils import *
from validation_fn import validation_fn
def alpha_weight(step, T1=100, T2=700, af=3):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af
     
def semisup_training_fn(train_loader, unlabeled_loader, val_loader, model, epochs, optimizer, loss_fn, saved_dir, resume_train, device):
    logging.info("....................Training ...................")
    best_score = 0
    last_epoch = 1
    step = 100 
    num_iter = 100
    
    
    # resume training
    saved_dir = os.path.join(saved_dir, saved_dir.split("/")[-1]+".pt")
    
    assert os.path.exists(saved_dir) is False
        
    if os.path.exists(saved_dir):
        last_epoch, train_f1, val_f1 = load_checkpoint(torch.load(saved_dir), model)
        logging.info("Loading checkpoint and continue tranining:{}".format(saved_dir))
        logging.info('Epoch: {0}: , Training F1_score: {1} , Val F1_score: {2}'.format(last_epoch, train_f1, val_f1))
        best_score = val_f1
        
    for epoch in range(last_epoch, epochs+1):
        loss_train_total = AverageMeter()
        logging.info('Epoch:{}'.format(epoch))
        predictions, true_labels = [], []
        
        total_unlabeled_loss = []
        progress_bar = tqdm(unlabeled_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for idx, batch in enumerate(progress_bar):
            image_input = batch["image_input"].to(device)
            diagnose_input = batch["diagnose_input"].to(device)
            bboxes_input = batch["bbox_input"].to(device)
            doctor_input = batch["doctor_input"].to(device)
            quantity_input = batch["quantity_input"].to(device)
            drugnames_input = batch["drugnames_input"].to(device)
            # targets = batch["targets"].to(device)
            model.eval()
            output_unlabeled = model(image_input, diagnose_input, drugnames_input, bboxes_input, doctor_input, quantity_input)
            output_unlabeled = torch.argmax(output_unlabeled, dim=1).cpu().numpy()
            
            model.train()
            output = model(image_input, diagnose_input, drugnames_input, bboxes_input, doctor_input, quantity_input)
            unlabeled_loss = loss_fn(output, output_unlabeled)
            unlabeled_loss = alpha_weight(step) * unlabeled_loss
            total_unlabeled_loss += unlabeled_loss
            
            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            
            if idx % num_iter == 0:
                # Normal training procedure
                total_labeled_loss = []
                progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
                for j, batch in enumerate(progress_bar):
                    image_input = batch["image_input"].to(device)
                    diagnose_input = batch["diagnose_input"].to(device)
                    bboxes_input = batch["bbox_input"].to(device)
                    doctor_input = batch["doctor_input"].to(device)
                    quantity_input = batch["quantity_input"].to(device)
                    drugnames_input = batch["drugnames_input"].to(device)
                    targets = batch["targets"].to(device)
                    
                    preds = model(image_input, diagnose_input, drugnames_input, bboxes_input, doctor_input, quantity_input)
                    labeled_loss = loss_fn(preds, targets)
                    total_labeled_loss += labeled_loss
                    
                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()
                    
                step += 1
                logging.info('labeled_loss: {:.3f}'.format(total_labeled_loss/int((j+1)*len(batch))))
                
        logging.info('unlabeled_loss: {:.3f}'.format(total_unlabeled_loss/int((idx+1)*len(batch))))
        logging.info("....................Validation ...................")
        val_acc, val_f1, val_loss = validation_fn(val_loader, model, loss_fn, epoch, device)
        logging.info("validation loss: {:.3f}".format(val_loss))
        logging.info("\t Validation accuracy score: {}".format(val_acc))
        logging.info("\t Validation f1 score: {}".format(val_f1))
        if val_f1 > best_score:
            best_score = val_f1
            checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_f1_score': train_f1, 'val_f1_score': val_f1}
            save_checkpoint(checkpoint, saved_dir)
            logging.info("Saving new checkpoint at {}".format(saved_dir))