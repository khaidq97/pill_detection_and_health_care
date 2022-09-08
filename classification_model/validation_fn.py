from tqdm import tqdm
import torch
import numpy as np

from utils import cls_metrics

def validation_fn(val_loader, model, loss_fn, epoch, device):
    model.eval()
    loss_val_total = 0
    predictions, true_labels = [], []

    progress_bar = tqdm(val_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for idx, batch in enumerate(progress_bar):
        with torch.no_grad():
            image_input = batch["image_input"].to(device)
            diagnose_input = batch["diagnose_input"].to(device)
            bboxes_input = batch["bbox_input"].to(device)
            doctor_input = batch["doctor_input"].to(device)
            quantity_input = batch["quantity_input"].to(device)
            drugnames_input = batch["drugnames_input"].to(device)
            targets = batch["targets"].to(device)

            preds = model(image_input, diagnose_input, drugnames_input, bboxes_input, doctor_input, quantity_input)

            loss = loss_fn(preds, targets)
            loss_val_total += loss.item()

            # softmax_fn = torch.nn.Softmax(dim=1)
            # preds = softmax_fn(preds)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            targets = torch.argmax(targets, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(targets)

    acc, f1 = cls_metrics(true_labels, predictions)
    val_loss = loss_val_total/int((idx+1))
    return acc, f1, val_loss