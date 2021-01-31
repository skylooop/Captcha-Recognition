import os
import glob
import numpy as np
from numpy.lib import utils
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import optim
import config
from torch.utils.data.dataloader import DataLoader
import dataset
from model import CaptchaModel
from torch.optim import Adam
import utils

def load():
    img_files = glob.glob(os.path.join(config.data_dir, "*.png"))
    labels_orig = [x.split('/')[-1][:-4] for x in img_files]
    labels = [[char for char in x] for x in labels_orig]  # all len 5
    lab_flat = [char for clist in labels for char in clist]
    encoder = LabelEncoder()
    encoder.fit_transform(lab_flat)
    targets_enc = [encoder.transform(x) for x in labels]
    targets_enc = np.array(targets_enc) + 1

    train_imgs, test_imgs, train_lab, test_lab, train_orig_lab, test_orig_lab = train_test_split(
        img_files, targets_enc, labels_orig, test_size=0.1)

    train_ds = dataset.Classification(
        train_imgs, targets=train_lab, resize=(config.h, config.w))

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=config.batch_size, num_workers=config.workers)
    
    test_ds = dataset.Classification(
        test_imgs, targets=test_lab, resize=(config.h, config.w))
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=config.batch_size, num_workers=config.workers)
    
    model = CaptchaModel(len(encoder.classes_)).to(config.device)
    optimizer = Adam(model.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 5, verbose = True)
    for epoch in range(10):
        train_loss = utils.train(model, train_loader, optimizer)
        valid_pred, valid_loss = utils.eval(model, train_loader)
        print(f"Epoch: {epoch}, train_loss: {train_loss}, val_loss: {valid_loss}")

load()