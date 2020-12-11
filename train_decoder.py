import torch
import sys
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import MRIIMGDataset, IRMDataset, IMGDataset
from models import Decoder
from losses import MRILoss, IMGLoss
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(datetime.now())
print("Working on {}.".format(device))
#sys.stdout.flush()

ID = sys.argv[1]
tmpdir = sys.argv[2]
with open(sys.argv[3]) as param_file:
    params = json.load(param_file)

out_path = "trained_models/decoder/decoder_{}/decoder".format(ID)
os.mkdir("trained_models/decoder/decoder_{}".format(ID)

## PARAMS ##
ENCODER_PATH = "trained_models/encoder/encoder14023375.pt"
FRIENDS_DIR = os.path.join(tmpdir, "fmri")
HCPTRT_DIR = os.path.join(tmpdir, "hcptrt")
N_VOXELS = 7435
FRAMES_DIR = os.path.join(tmpdir, "frames/data/preprocessed/frames")
IMAGENET_PATH = os.path.join(tmpdir, "Imagenet64val_data")
FILTERS_TNG = params["filters_tng"]
FILTERS_VAL = params["filters_val"]
BS = params["batch_size"]
NB_EPOCHS = params["nb_epochs"]
######################################
# TODO: make params
LR = 0.1
L1_conv = 1e5
L1_fc = 10
L2 = 0.001

#print("np epochs", NB_EPOCHS, "; BS", BS, "; LR", LR, "; L1_conv", L1_conv, "; L1_fc", L1_fc, "; L2", L2) 


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

dataset_mriimg_tng = MRIIMGDataset(FRIENDS_DIR, FRAMES_DIR, FILTERS_MRIIMG_TNG)
dataset_mriimg_val = MRIIMGDataset(FRIENDS_DIR, FRAMES_DIR, FILTERS_MRIMG_VAL)
dataset_img_tng = IMGDataset(IMAGENET_PATH, validation=False)
dataset_img_val = IMGDataset(IMAGENET_PATH, validation=True)
dataset_mri_tng = MRIDataset(HCPTRT_PATH, validation=False)
dataset_mri_val = MRIDataset(HCPTRT_PATH, validation=True)

dataloader_mriimg_tng = DataLoader(dataset_mriimg_tng, batch_size=BS, shuffle=True, num_workers=4)
dataloader_mriimg_val = DataLoader(dataset_mriimg_val, batch_size=BS, shuffle=True, num_workers=4)
dataloader_img_tng = DataLoader(dataset_img_tng, batch_size=BS, shuffle=True, num_workers=4)
dataloader_img_val = DataLoader(dataset_img_val, batch_size=BS, shuffle=True, num_workers=4)
dataloader_mri_tng = DataLoader(dataset_mri_tng, batch_size=BS, shuffle=True, num_workers=4)
dataloader_mri_val = DataLoader(dataset_mri_val, batch_size=BS, shuffle=True, num_workers=4)


encoder = torch.load(ENCODER_PATH).to(device)
encoder.eval()
decoder = Decoder(N_VOXELS).to(device)
optimizer = optim.Adam(decoder.parameters(), lr=LR, weight_decay=L2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_decay)
loss_MRI = MRILoss(device)
loss_IMG = IMGLoss(device)

losses_mriimg_tng = []
losses_mriimg_val = []
losses_irm_tng = []
losses_irm_val = []
losses_img_tng = []
losses_img_val = []


## TRAINING LOOP ##
for epoch in range(NB_EPOCHS):
    print("epoch", epoch)

    ## MRIIMG
    decoder.train()
    loss_sum_tng = 0.
    for i_batch, sampled_batch in enumerate(dataloader_mriimg_tng):
        optimizer.zero_grad()
        img = sampled_batch["img"].to(device)
        mri = sampled_batch["mri"].to(device)
        pred_img = decoder(mri)
        loss = loss_IMG(pred_img, img) #TODO: add difference of VGG features to loss
        loss.sum().backward()
        optimizer.step()
        loss_sum_tng += loss.sum().item()/BS/len(dataloader_mriimg_tng)
    losses_mriimg_tng.append(loss_sum_tng)
    scheduler.step()

    loss_sum_val = 0.
    decoder.eval()
    for i_batch, sampled_batch in enumerate(dataloader_val):
        pred_img = decoder(sampled_batch["mri"].to(device))
        img = sampled_batch["img"].to(device)
        loss = loss_IMG(pred_img, img)
        loss_sum_val += loss.sum().item()/BS/len(dataloader_mriimg_val)
    losses_mriimg_val.append(loss_sum_val)

    print("mriimg", "tng:", loss_sum_tng, "val:", loss_sum_val)

    ## IMG
    decoder.train()
    loss_sum_tng = 0.
    for i_batch, sampled_batch in enumerate(dataloader_img_tng):
        optimizer.zero_grad()
        img = sampled_batch["img"].to(device)
        pred_img = decoder(encoder(img))
        loss = loss_IMG(pred_img, img) #TODO: add difference of VGG features to loss
        loss.sum().backward()
        optimizer.step()
        loss_sum_tng += loss.sum().item()/BS/len(dataloader_img_tng)
    losses_img_tng.append(loss_sum_tng)

    loss_sum_val = 0.
    decoder.eval()
    for i_batch, sampled_batch in enumerate(dataloader_img_val):
        img = sampled_batch["img"].to(device)
        pred_img = decoder(encoder(img))
        loss = loss_IMG(pred_img, img)
        loss_sum_val += loss.sum().item()/BS/len(dataloader_img_val)
    losses_img_val.append(loss_sum_val)

    print("img   ", "tng:", loss_sum_tng, "val:", loss_sum_val)

    ## MRI
    decoder.train()
    loss_sum_tng = 0.
    for i_batch, sampled_batch in enumerate(dataloader_mri_tng):
        optimizer.zero_grad()
        mri = sampled_batch["mri"].to(device)
        pred_mri = encoder(decoder(mri))
        loss = loss_MRI(pred_mri, mri)
        loss.sum().backward()
        optimizer.step()
        loss_sum_tng += loss.sum().item()/BS/len(dataloader_mri_tng)
    losses_mri_tng.append(loss_sum_tng)

    loss_sum_val = 0.
    decoder.eval()
    for i_batch, sampled_batch in enumerate(dataloader_mri_val):
        mri = sampled_batch["mri"].to(device)
        pred_mri = decoder(encoder(mri))
        loss = loss_IMG(pred_mri, mri)
        loss_sum_val += loss.sum().item()/BS/len(dataloader_mri_val)
    losses_mri_val.append(loss_sum_val)

    print("mri   ", "tng:", loss_sum_tng, "val:", loss_sum_val)

    scheduler.step()


torch.save(encoder, out_path+"_"+ID+".pt")
np.save(out_path+"_"+ID+"_loss_mriimg_tng.npy", np.array(losses_mriimg_tng))
np.save(out_path+"_"+ID+"_loss_mriimg_val.npy", np.array(losses_mriimg_val))

np.save(out_path+"_"+ID+"_loss_img_tng.npy", np.array(losses_img_tng))
np.save(out_path+"_"+ID+"_loss_img_val.npy", np.array(losses_img_val))

np.save(out_path+"_"+ID+"_loss_mri_tng.npy", np.array(losses_mri_tng))
np.save(out_path+"_"+ID+"_loss_mri_val.npy", np.array(losses_mri_val))
