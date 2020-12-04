import numpy as np
import torch
import os
import PIL
from torchvision import transforms
from torch.utils.data import Dataset


img_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MRIImgDataset(Dataset):
    def __init__(self, mri_dir, img_dir, filters=[], to_tensor=True):

        mri_path_list = [os.path.join(mri_dir, p) for p in os.listdir(mri_dir) if ".npy" in p]
        img_path_list = [os.path.join(img_dir, p) for p in os.listdir(img_dir) if ".jpeg" in p]

        for filter in filters:
            mri_path_list = [p in mri_path if filter in p]
            img_path_list = [p in img_path if filter in p]

        self.handlers = []
        for mri_path in mri_path_list:
            mri_data = np.load(mri_path)
            if to_tensor:
                mri_data = torch.tensor(mri_data, dtype=torch.float32)
            episode = os.path.split(mri_path)[1].split("task-")[1].split("_")[0]
            for img_path in img_path_list:
                if episode in img_path:
                    t = int(os.path.split(img_path)[0].split("t-")[1].split(".")[0])
                    img_data = PIL.Image.open(img_path)
                    if to_tensor:
                        img_data = img_transform(img_data).type(torch.float32)
                    self.handlers.append({"mri":mri_data[t], "img":img_data})


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]


class MRIDataset(Dataset):
    def __init__(self, mri_dir, filters=[], to_tensor=True):

        mri_path_list = [os.path.join(mri_dir, p) for p in os.listdir(mri_dir) if ".npy" in p]

        for filter in filters:
            mri_path_list = [p in mri_path if filter in p]

        self.handlers = []
        for mri_path in mri_path_list:
            mri_data = np.load(mri_path)
            if to_tensor:
                mri_data = torch.tensor(mri_data, dtype=torch.float32)
            for t in range(mri_data.shape[0])
                self.handlers.append(mri_data[t])


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]


class IMGDataset(Dataset):
    def __init__(self, img_dir, filters=[], to_tensor=True):

        img_path_list = [os.path.join(img_dir, p) for p in os.listdir(img_dir) if ".npy" in p]

        for filter in filters:
            img_path_list = [p in img_path if filter in p]

        self.handlers = []
        for img_path in img_path_list:
            img_data = PIL.Image.open(img_path)
            if to_tensor:
                img_data = img_transform(img_data).type(torch.float32)
            self.handlers.append(img_data)


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        return self.handlers[index]
