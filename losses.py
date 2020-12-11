from torch.nn import MSELoss, CosineSimilarity

class MRILoss(object):
    def __init__(self, device, k_cosine=0.1):
        self.k = k_cosine
        self.mse = MSELoss().to(device)
        self.cosine_sim = CosineSimilarity().to(device)

    def __call__(self, x, y):
        loss = self.mse(x,y)
        if self.k != 0:
            loss = loss - self.k * self.cosine_sim(x, y) 
        return loss