import numpy as np
import torch
import os
import pickle
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


torch_transform = transforms.Compose([
    #transforms.Resize(112),
    #transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def img_transform(img):
    w, h = img.size
    n_w = int(w/h*112)
    left = (n_w- 112)/2
    right = (n_w + 112)/2
    img = img.resize((n_w,112)).crop((left,0,right,112))
    return torch_transform(img)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MRIIMGDataset(Dataset):
    def __init__(self, mri_dir, img_dir, filters=[], delay=5.4,
                 fps=4.994742376, tr=1.49, img_per_mri=1):

        mri_path_list = [os.path.join(mri_dir, p) for p in os.listdir(mri_dir) if ".npy" in p]

        for filter in filters:
            mri_path_list = [p for p in mri_path_list if filter in p]

        self.handlers = []
        for mri_path in mri_path_list:
            mri_data = np.load(mri_path)
            mri_data = torch.tensor(mri_data, dtype=torch.float32)
            episode = os.path.split(mri_path)[1].split("task-")[1].split("_")[0]
            for i in range(mri_data.shape[0]):
                i_img_peak = int((i*tr-delay)*fps)
                i_img_start = i_img_peak - int(img_per_mri/2)
                if i_img_start > 0:
                    imgs = []
                    for i_img in range(i_img_start, i_img_start+img_per_mri):
                        img_path = os.path.join(img_dir, episode, episode+"_frame-{}.jpg".format(i_img))
                        img_data = PIL.Image.open(img_path)
                        img_data = img_transform(img_data).type(torch.float32)
                        imgs.append(img_data)
                    imgs = torch.stack(imgs)
                    self.handlers.append({"mri":mri_data[i], "img":imgs})


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]


class MRIDataset(Dataset):
    def __init__(self, mri_dir, filters=[], ratio=0.8, validation=False):

        mri_path_list = [os.path.join(mri_dir, p) for p in os.listdir(mri_dir) if ".npy" in p]

        for filter in filters:
            mri_path_list = [p for p in mri_path if filter in p]

        self.handlers = []
        if validation:
            i_start = int(ratio*len(mri_path_list))
            i_end = len(mri_path_list)
        else:
            i_start = 0
            i_end = int(ratio*len(mri_path_list))
        for i_mri_path in range(i_start, i_end):
            mri_data = np.load(mri_path_list[i_mri_path])
            mri_data = torch.tensor(mri_data, dtype=torch.float32)
            for t in range(mri_data.shape[0]):
                self.handlers.append({"mri":mri_data[t]})


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]


class IMGDataset(Dataset):
    def __init__(self, imagenet_path, ratio=0.8, validation=False):

        imgdict = unpickle(imagenet_path)
        self.handlers = []
        if validation:
            i_start = int(ratio*len(imgdict["data"]))
            i_end = len(imgdict["data"])
        else:
            i_start = 0
            i_end = int(ratio*len(imgdict["data"]))
        for i_img in range(i_start, i_end):
            img = np.moveaxis(imgdict['data'][i_img].reshape(3, 64, 64).astype(np.uint8), 0, 2)
            img = Image.fromarray(img).resize((112,112))
            img = torch.stack([torch_transform(img)])
            self.handlers.append({"img":img})


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]
