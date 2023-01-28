import torch
import torch.nn as nn
from torch.distributions import Beta

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, Y_mask=None,weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        #BxC
        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]

        #BxCxW
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        
        #BxCxHxW
        elif n_dims == 4:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        #BxTxCxHxW
        else:
            X = coeffs.view(-1, 1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1, 1)) * X[perm]


        #BxCxHxW for segmentation / change it for other purposes
        Y = coeffs.view(-1, 1, 1, 1) * Y + (1 - coeffs.view(-1, 1, 1, 1)) * Y[perm]

        #X = torch.nan_to_num(X,nan=0.0)

        ret = [X,Y]
        
        if Y_mask is not None:
            Y_mask = Y_mask | Y_mask[perm]
            ret.append(Y_mask)

        if weight is not None:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            ret.append(weight)
        
        
        return ret