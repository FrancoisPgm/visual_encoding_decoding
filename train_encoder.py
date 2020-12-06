import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIImgDataset
from models import Encoder
from loss import MRILoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print "Working on {}.".format(device)

out_path = "trained_models/encoder"

## PARAMS ##
MRI_DIR = "data/preprocessed/mri"
N_VOXELS = 7435
IMG_DIR = "data/preprocessed/frames"
FILTERS_TNG = ["sub-01", "task-s01"]
FILTERS_VAL = ["sub-01", "task-s02"]
BS = 15
NB_EPOCHS = 100
LR = 0.1
L1 = 0.1
L2 = 0.1

def step_decay(epoch):
    lrate = 1
    if(epoch>20):
        lrate = 0.1
    if (epoch > 35):
        lrate = 0.01
    if (epoch > 45):
        lrate = 0.001
    if (epoch > 50):
        lrate = 0.0001
    return lrate

dataset_tng = MRIImgDataset(MRI_DIR, IMG_DIR, FILTERS_TNG)
dataset_val = MRIImgDataset(MRI_DIR, IMG_DIR, FILTERS_VAL)

dataloader_tng = DataLoader(dataset_tng, batch_size=BS, shuffle=True, num_workers=1)
dataloader_val = DataLoader(dataset_val, batch_size=BS, shuffle=True, num_workers=1)

encoder = Encoder().to(device)
optimizer = optim.Adam(encoder.parameters(), lr=LR, weight_decay=L2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_decay)
loss_function = MRILoss(device)
l1_penalty = torch.nn.L1Loss().to(device)

losses_tng = []
losses_val = []

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
            reg_loss += l1_penalty(param)
        tot_loss = loss + L1 * reg_loss
        tot_loss.backward()
        optimizer.step()
        loss_sum_tng += loss.item()/len(dataloader_tng)
    losses_tng.append(loss_sum_tng)
    scheduler.step()

    loss_sum_val = 0.
    encoder.eval()

    for i_batch, sampled_batch in enumerate(dataloader_val):
        output = encoder(sampled_batch["img"].to(device))
        gts = sampled_batch["mri"]
        loss = loss_function(output, gts)
        loss_sum_val += loss.item()/len(dataloader_val)
    losses_val.append(loss_sum_val)

    print("epoch", epoch, "tng:", loss_sum_tng, "val:", loss_sum_val)

torch.save(encoder, out_path+".pt")
np.save(out_path+"_loss_tng.npy", np.array(losses_tng))
np.save(out_path+"_loss_val.npy", np.array(losses_val))
