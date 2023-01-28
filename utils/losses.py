
import torch.nn as nn
import torch
import torch.nn.functional as F


from segmentation_models_pytorch.losses import DiceLoss,SoftBCEWithLogitsLoss,TverskyLoss

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _soft_neg_loss(pred, gt,eps=0.2):
  ''' Modified focal loss to soft focal loss for noisy labels :)
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  mask = gt.ge(0).float() * gt.le(1).float()

  pos_inds = gt.ge(1 - eps).float() * mask
  neg_inds = gt.lt(1 - eps).float() * mask

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class KeypointFocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  __name__ = 'keypoint_focal_loss'
  def __init__(self):
    super(KeypointFocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    out = out.clamp_max(0.999).clamp_min(0.0001)
    return self.neg_loss(out, target)

class SmoothKeypointFocalLoss(nn.Module):
  '''nn.Module warpper for smooth focal loss'''
  __name__ = 'keypoint_smooth_focal_loss'
  def __init__(self,eps):
    super(SmoothKeypointFocalLoss, self).__init__()
    self.eps = eps
    self.soft_neg_loss = _soft_neg_loss

  def forward(self, out, target):
    out = out.clamp_max(0.999).clamp_min(0.0001)
    return self.soft_neg_loss(out, target,self.eps)


class DiceBCELoss(nn.Module):

  def __init__(self,alpha_beta=(1.0,1.0),mode='binary',bce_pos_weight=1.0):
    super().__init__()
    
    self.dice = DiceLoss(mode=mode,from_logits=True,smooth=0,eps=1e-7)
    self.bce = SoftBCEWithLogitsLoss(pos_weight=bce_pos_weight)

    self.alpha ,self.beta = alpha_beta
  
  def forward(self,pred,gt):

    dice_loss = self.dice(pred,gt)
    bce_loss = self.bce(pred,gt)

    loss = self.alpha * dice_loss + self.beta * bce_loss

    return loss
 


class MaskedBCELoss(nn.Module):

  def __init__(self,device='cpu'):
    super().__init__()
    self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
    self.device = device 
  
  def forward(self,pred,gt,mask):
    B,C,H,W = pred.shape
    loss = torch.zeros_like(pred, requires_grad=True).to(device=self.device)

    pred = torch.masked_select(pred,mask)
    gt = torch.masked_select(gt,mask)

    if pred.shape[0] != 0:
      bce_loss = self.bce_criterion(pred,gt)
      loss = loss.masked_scatter(mask,bce_loss)

    loss = loss.view(B,C,-1)
    mask = mask.view(B,C,-1)
    counts = mask.float().sum(dim=-1).clamp_min(1.0)

    loss = loss.sum(dim=-1) / counts
    loss = loss.mean()

    return loss







if __name__ == '__main__':

  #"""
  x2 = torch.tensor(
    [
      [
        [[0,0.2,0.3],
          [0.2,0.1,0.1],
          [0.3,0.5,0.5]]
      ]
    ])

  x = torch.tensor(
    [
      [
        [[0,1.0,1.0],
        [1.0,1.0,0.0],
        [1.0,0.0,0.0]]
      ]
    ])
  
  
  mask = (x != 0).bool()
  crit = MaskedBCELoss()
  print(x.shape)
  crit(x2,x,mask)
  print(x.shape)

  #pos = x.gt(2)
  #print(pos.shape)
  #print(pos)
  #"""