import torch
import sys
import os
import json
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import MRIIMGDataset
from models import Encoder
from losses import MRILoss
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(datetime.now())
print("Working on {}.".format(device))
#sys.stdout.flush()
out_path = "trained_models/encoder/encoder"

ID = sys.argv[1]
tmpdir = sys.argv[2]
with open(sys.argv[3]) as param_file:
    params = json.load(param_file)

## PARAMS ##
MRI_DIR = os.path.join(tmpdir, "fmri")
N_VOXELS = 7435
IMG_DIR = os.path.join(tmpdir, "frames/data/preprocessed/frames")
FILTERS_TNG = params["filters_tng"]# ["sub-0", "task-s01"]
FILTERS_VAL = params["filters_val"]#["sub-0", "task-s02"]
BS = params["batch_size"]#100
NB_EPOCHS = params["nb_epochs"]#100
LR = params["lr"]#0.01
L1_conv = params["L1_conv"]#0.0001
L1_fc = params["L1_fc"]#0.0001 
L2 = params["L2"]#0.0001
k_cosine = params["k_cosine"]#0.1
dropout = params["dropout"]

if params["tag"]:
    out_path += "_"+params["tag"]

print("filters tng", FILTERS_TNG)
print("filters val", FILTERS_VAL)
print("np epochs", NB_EPOCHS, "; BS", BS, "; LR", LR, "; L1_conv", L1_conv, "; L1_fc", L1_fc, "; L2", L2, ": dropout", dropout, "; k_cosine", k_cosine) 


def step_decay(epoch):
    lrate = 1
#    if(epoch > NB_EPOCHS/4):
#        lrate = 0.1
#    if (epoch > 1.5*NB_EPOCHS/4):
#        lrate = 0.01
#    if (epoch > NB_EPOCHS/2):
#        lrate = 0.001
#    if (epoch > 3*NB_EPOCHS/4):
#        lrate = 0.0001
    return lrate

dataset_tng = MRIIMGDataset(MRI_DIR, IMG_DIR, FILTERS_TNG)
print(datetime.now(), "tng dataset done")
dataset_val = MRIIMGDataset(MRI_DIR, IMG_DIR, FILTERS_VAL)

print(datetime.now(), "val dataset done")
#sys.stdout.flush()

dataloader_tng = DataLoader(dataset_tng, batch_size=BS, shuffle=True, num_workers=8)
print(datetime.now(), "tng dataloader done", len(dataloader_tng))
dataloader_val = DataLoader(dataset_val, batch_size=BS, shuffle=True, num_workers=8)
print(datetime.now(), "val dataloader done", len(dataloader_val))

encoder = Encoder(dropout=dropout).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=LR, weight_decay=L2)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
loss_function = MRILoss(device, k_cosine=k_cosine)

losses_tng = []
losses_val = []
examples_pred = []
examples_gt = []

#print(datetime.now(), "begin training")
#sys.stdout.flush()

## TRAINING LOOP ##
for epoch in range(NB_EPOCHS):
    encoder.train()
    loss_sum_tng = 0.
    for i_batch, sampled_batch in enumerate(dataloader_tng):
        optimizer.zero_grad()
        input = sampled_batch["img"].to(device)
        output = encoder(input)
        gts = sampled_batch["mri"]
        loss = loss_function(output, gts.to(device))
        reg_loss = 0
        for param in encoder.parameters():
            if len(param.shape)>1:
                if param.shape[0] == N_VOXELS:
                    l1_factor = optimizer.param_groups[0]['lr'] * L1_fc
                else:
                    l1_factor = optimizer.param_groups[0]['lr'] * L1_conv
                reg_loss += l1_factor * torch.norm(param, 1)
        tot_loss = loss + reg_loss
        tot_loss.sum().backward()
        optimizer.step()
        loss_sum_tng += loss.sum().item()/len(dataloader_tng)
    losses_tng.append(loss_sum_tng)
    if not epoch%10:
        examples_gt.append(gts[0].to('cpu').detach().numpy())
        examples_pred.append(output[0].to('cpu').detach().numpy())

    scheduler.step(loss_sum_tng)

    loss_sum_val = 0.
    encoder.eval()

    for i_batch, sampled_batch in enumerate(dataloader_val):
        output = encoder(sampled_batch["img"].to(device))
        gts = sampled_batch["mri"]
        loss = loss_function(output, gts.to(device))
        loss_sum_val += loss.sum().item()/len(dataloader_val)
    losses_val.append(loss_sum_val)

    #print(datetime.now())
    print("epoch", epoch, "tng:", loss_sum_tng, "val:", loss_sum_val)
    #sys.stdout.flush()

torch.save(encoder.to('cpu'), out_path+"_"+ID+".pt")
np.save(out_path+"_"+ID+"_examples_pred.npy", np.array(examples_pred))
np.save(out_path+"_"+ID+"_examples_gt.npy", np.array(examples_gt))
#np.save(out_path+"_"+ID+"_loss_tng.npy", np.array(losses_tng))
#np.save(out_path+"_"+ID+"_loss_val.npy", np.array(losses_val))
