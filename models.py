import torch
import torch.nn as nn
from datasets import img_transform


def normal_init(m, mean=0., std=1.):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, n_voxels=7435, dropout=0.2, n_img=1):
        super(Encoder, self).__init__()
        self.n_img = n_img
        self.n_voxels=n_voxels
        self.preprocess = img_transform
        self.alexnetConv1 = nn.Sequential(
                torch.load("data/external/alexconv1.pkl"),
                nn.ReLU(),
                #nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 32, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
        )
        self.flatten = nn.Flatten(start_dim=1)
        if self.n_img > 1:
            self.conv_temp = nn.Conv1d(n_img, 1, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1152, n_voxels)
        self.weight_init()

    def forward(self, x):
        if self.n_img > 1:
            y = [self.alexnetConv1(x[:,i]) for i in range(x.shape[1])]
            y = [self.conv2(z) for z in y]
            y = [self.conv3(z) for z in y]
            y = torch.stack(y, dim=1)
            y = y.view(y.shape[0], y.shape[1], -1)
            y = self.conv_temp(y).squeeze(dim=1)
        else:
            x = x[:,0]
            y = self.alexnetConv1(x)
            y = self.conv2(y)
            y = self.conv3(y)
            y = self.flatten(y)
        y = self.dropout(y)
        y = self.fc(y)
        return y

    def predict(self, x):
        self.eval()
        if n_img > 1:
            x = torch.stack([self.preprocess(z) for z in x])
        else:
            x = self.preprocess(x).unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        y = self.forward(x).squeeze()
        return y

    def weight_init(self):
        for block in self._modules:
            if block != "alexnetConv1":
                try:
                    for m in self._modules[block]:
                        normal_init(m,mean,std)
                except:
                    normal_init(block)



class Decoder(nn.Module):
    def __init__(self, n_voxels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(n_voxels, 14*14*64)
        self.conv1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(64, 3, 3, stride=1, padding=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        y = y.view((-1, 64, 14, 14))
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y

    def predict(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = self.forward(x).squeeze()
        return y

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m,mean,std)
            except:
                normal_init(block)



#def VAE(nn.Module):
#    def __init__(self, n_voxels, n_latent=1024):
#        self.conv1 = nn.Sequential(
#                nn.Conv2d(3, 128, 3, stride=2, padding=2),
#                nn.ReLU(),
#                nn.BatchNorm2d(128)
#                )
#        self.conv2 = nn.Sequential(
#                nn.Conv2d(128, 256, 3, stride=2, padding=2),
#                nn.ReLU(),
#                nn.BatchNorm2d(256)
#                )
#        self.conv3 = nn.Sequential(
#                nn.Conv2d(256, 512, 3, stride=2, padding=2),
#                nn.ReLU(),
#                nn.BatchNorm2d(512)
#                )
#        self.conv4 = nn.Sequential(
#                nn.Conv2d(512, 1024, 3, stride=2, padding=2),
#                nn.ReLU(),
#                nn.BatchNorm2d(1024)
#                )
#        self.conv5 = nn.Sequential(
#                nn.Conv2d(512, 256, 3, stride=2, padding=2),
#                nn.ReLU(),
#                nn.BatchNorm2d(256),
#                nn.Flatten()
#                )
#
#        self.deconv1 = nn.Sequential(



