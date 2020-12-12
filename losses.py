import torch
from torch.nn import MSELoss, CosineSimilarity

class MRILoss(object):
    def __init__(self, device, k_cosine=0.1):
        self.k = k_cosine
        self.mse = MSELoss().to(device)
        self.cosine_sim = CosineSimilarity().to(device)

    def __call__(self, x, y):
        loss = self.mse(x,y) - self.k * self.cosine_sim(x, y)
        return loss


class IMGLoss(object):
    def __init__(self, device, k_L1, k_VGG, k_TV):
        self.k_L1 = k_L1
        self.k_VGG = k_VGG
        self.k_TV = k_TV
        self.VGG19Conv1 = torch.load("data/external/vgg19conv1.pkl").to(device)
        self.VGG19Conv1.eval()
        self.VGG19Conv2 = torch.load("data/external/VGG19Conv2.pkl").to(device)
        self.VGG19Conv2.eval()

    def TV_loss(self, img):
         bs_img, c_img, h_img, w_img = img.size()
         tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
         tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
         return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def __call__(self, pred_img, gt_img):
        feat_pred_1 = self.VGG19Conv1(pred_img)
        feat_gt_1 = self.VGG19Conv1(gt_img)
        feat_pred_2 = self.VGG19Conv2(feat_pred_1)
        feat_gt_2 = self.VGG19Conv2(feat_gt_1)
        L1_loss = self.k_L1 * torch.norm(pred_img-gt_img, 1)
        VGG_loss = self.k_VGG * (torch.norm(feat_pred_1-feat_gt_1, 2)+torch.norm(feat_pred_2-feat_gt_2, 2))
        TV_loss =  self.k_TV * self.TV_loss(pred_img)
        loss = L1_loss + VGG_loss + TV_loss
        return loss
