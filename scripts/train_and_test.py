from __future__ import print_function, division

import os
import sys
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import json
import importlib
from glob import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from losses import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss

from utils import (
    show_image_and_mask,
    load_config,
    _print,
)


from datasets.base_dataset import segmentation_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#torch.manual_seed(0)
#np.random.seed(0)
#torch.cuda.manual_seed(0)
#import random
#random.seed(0)

CONFIG_NAME = "nuclei_uctransnet.yaml"
CONFIG_FILE_PATH = os.path.join("../configs", CONFIG_NAME)

config = load_config(CONFIG_FILE_PATH)
_print("Config:", "info_underline")
print(json.dumps(config, indent=2))
print(20*"~-", "\n")


INPUT_SIZE = config['dataset']['input_size']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {device}")


def prepare_datasets(config):
    image_files = config['dataset']['training']['params']['image_list']
    mask_files = config['dataset']['training']['params']['mask_list']
    train_dataset = segmentation_dataset(image_files, mask_files, mode="train", one_hot=True)
    val_dataset = segmentation_dataset(image_files, mask_files, mode="val", one_hot=True)
    test_dataset = segmentation_dataset(image_files, mask_files, mode="test", one_hot=True)

    tr_datalodaer = DataLoader(train_dataset, **config['data_loader']['train'])
    val_dataloader = DataLoader(val_dataset, **config['data_loader']['validation'])
    test_dataloader = DataLoader(test_dataset, **config['data_loader']['test'])

    return tr_datalodaer, val_dataloader, test_dataloader


def get_metrics(num_classes):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(task="multiclass", num_classes=2),
            torchmetrics.Accuracy(task="multiclass", num_classes=2),
            torchmetrics.Dice(),
            torchmetrics.Precision(task="multiclass", num_classes=2),
            torchmetrics.Specificity(task="multiclass", num_classes=2),
            torchmetrics.Recall(task="multiclass", num_classes=2),
            # IoU
            torchmetrics.JaccardIndex(task="multiclass", num_classes=2)
        ],
        prefix='train_metrics/'
    )

    # train_metrics
    train_metrics = metrics.clone(prefix='train_metrics/').to(device)

    # valid_metrics
    valid_metrics = metrics.clone(prefix='valid_metrics/').to(device)

    # test_metrics
    test_metrics = metrics.clone(prefix='test_metrics/').to(device)
    
    return train_metrics, valid_metrics, test_metrics


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


def validate(model, criterion, vl_dataloader, valid_metrics):
    model.eval()
    with torch.no_grad():
        
        evaluator = valid_metrics.clone().to(device)
        
        losses = []
        cnt = 0.
        for batch, batch_data in enumerate(vl_dataloader):
            imgs = batch_data['image']
            msks = batch_data['mask']
            
            cnt += msks.shape[0]
            
            imgs = imgs.to(device)
            msks = msks.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, msks)
            losses.append(loss.item())
            
            
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)
        
#             _cml = f"curr_mean-loss:{np.sum(losses)/cnt:0.5f}"
#             _bl = f"batch-loss:{losses[-1]/msks.shape[0]:0.5f}"
#             iterator.set_description(f"Validation) batch:{batch+1:04d} -> {_cml}, {_bl}")
        
        # print the final results
        loss = np.sum(losses)/cnt
        metrics = evaluator.compute()
    
    return evaluator, loss


def train(
    model, 
    device, 
    tr_dataloader,
    vl_dataloader,
    config,
    
    criterion,
    optimizer,
    scheduler,
    train_metrics,
    valid_metrics,
    save_dir='./',
    save_file_id=None,
):
    
    EPOCHS = config['training']['epochs']
    
    torch.cuda.empty_cache()
    model = model.to(device)

    evaluator = train_metrics.clone().to(device)
    
    epochs_info = []
    best_model = None
    best_result = {}
    best_vl_loss = np.Inf
    for epoch in range(EPOCHS):
        model.train()
        
        evaluator.reset()
        tr_iterator = tqdm(enumerate(tr_dataloader))
        tr_losses = []
        cnt = 0
        for batch, batch_data in tr_iterator:
            imgs = batch_data['image']
            msks = batch_data['mask']
            
            imgs = imgs.to(device)
            msks = msks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, msks)
            loss.backward()
            optimizer.step()
            
            # evaluate by metrics
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)
            
            cnt += imgs.shape[0]
            tr_losses.append(loss.item())
            
            # write details for each training batch
            _cml = f"curr_mean-loss:{np.sum(tr_losses)/cnt:0.5f}"
            _bl = f"mean_batch-loss:{tr_losses[-1]/imgs.shape[0]:0.5f}"
            tr_iterator.set_description(f"Training) ep:{epoch:03d}, batch:{batch+1:04d} -> {_cml}, {_bl}")
            
        tr_loss = np.sum(tr_losses)/cnt
        
        # validate model
        vl_metrics, vl_loss = validate(model, criterion, vl_dataloader, valid_metrics)
        if vl_loss < best_vl_loss:
            # find a better model
            best_model = model
            best_vl_loss = vl_loss
            best_result = {
                'tr_loss': tr_loss,
                'vl_loss': vl_loss,
                'tr_metrics': make_serializeable_metrics(evaluator.compute()),
                'vl_metrics': make_serializeable_metrics(vl_metrics.compute())
            }
        
        # write the final results
        epoch_info = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'tr_metrics': make_serializeable_metrics(evaluator.compute()),
            'vl_metrics': make_serializeable_metrics(vl_metrics.compute())
        }
        epochs_info.append(epoch_info)
#         epoch_tqdm.set_description(f"Epoch:{epoch+1}/{EPOCHS} -> tr_loss:{tr_loss}, vl_loss:{vl_loss}")
        evaluator.reset()
    
        scheduler.step(vl_loss)
  
    # save final results
    res = {
        'id': save_file_id,
        'config': config,
        'epochs_info': epochs_info,
        'best_result': best_result
    }
    fn = f"{save_file_id+'_' if save_file_id else ''}result.json"
    fp = os.path.join(config['model']['save_dir'],fn)
    with open(fp, "w") as write_file:
        json.dump(res, write_file, indent=4)

    # save model's state_dict
    fn = "last_model_state_dict.pt"
    fp = os.path.join(config['model']['save_dir'],fn)
    torch.save(model.state_dict(), fp)
    
    # save the best model's state_dict
    fn = "best_model_state_dict.pt"
    fp = os.path.join(config['model']['save_dir'], fn)
    torch.save(best_model.state_dict(), fp)
    
    return best_model, model, res


def test(model, te_dataloader, test_metrics):
    model.eval()
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)            
        for batch_data in tqdm(te_dataloader):
            imgs = batch_data['image']
            msks = batch_data['mask']
            
            imgs = imgs.to(device)
            msks = msks.to(device)
            
            preds = model(imgs)
            
            # evaluate by metrics
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)
            
    return evaluator


