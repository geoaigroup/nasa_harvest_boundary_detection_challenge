import torch
import torch.nn as nn
#import pandas as pd


def _threshold(x, threshold=0.5):
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x


class MicroScores:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-8
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.
        self.tn = 0.

    def get_scores(self):
        eps = self.eps
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        recall =(tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        
        iou = (tp + eps) / (tp + fp + fn + eps)
        f1_score = ((2 * recall * precision) + eps) / (recall + precision + eps)

        return {
            'recall' : recall.mean().item(),
            'precision' : precision.mean().item(),
            'iou' : iou.mean().item(),
            'f1' : f1_score.mean().item()
        }

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):
        
        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)

        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW

        tp = torch.sum(y_pr * y_gt,dim=-1)
        tn = torch.sum((1.0 - y_pr) * (1.0 - y_gt),dim=-1)
        fp = torch.sum(y_pr,dim=-1) - tp
        fn = torch.sum(y_gt,dim=-1) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        return self.get_scores()

class DiceScore:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.dice_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):

        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW


        intersection = torch.sum(y_pr * y_gt, dim=-1)
        cardinality = torch.sum(y_pr + y_gt, dim=-1)

        dice_score = (2.0 * intersection)/ cardinality.clamp_min(self.eps)
        
        weights = 1.0 - (cardinality == 0.0).float()
        weights = weights.sum(dim=-1)

        dice_score = dice_score.sum(dim=-1) / weights.clamp_min(1.0)


        self.dice_score_sum += dice_score.sum()
        self.weights_sum += (weights > 0.0).float().sum()

        dice_score = self.dice_score_sum / self.weights_sum.clamp_min(1.0)


        return dice_score

class IoUScore:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.iou_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):


        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW


        intersection = torch.sum(y_pr * y_gt, dim=-1)
        union = torch.sum((y_pr + y_gt).clamp_max(1.0), dim=-1)

        iou_score =  intersection / union.clamp_min(self.eps)
        
        weights = 1.0 - (union == 0.0).float()
        weights = weights.sum(dim=-1)

        iou_score = iou_score.sum(dim=-1) / weights.clamp_min(1.0)


        self.iou_score_sum += iou_score.sum()
        self.weights_sum += (weights > 0.0).float().sum()

        iou_score = self.iou_score_sum / self.weights_sum.clamp_min(1.0)


        return iou_score


if __name__ == '__main__':
    pass
    """
    y_pred = torch.tensor(
        
        [[
            [[0,1,0],
             [0,1,0],
             [0,0,0]],
            [[0,1,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]]
        ],
         [
            [[0,1,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]]
        ],
        ]
        
    ).float()

    y_gt = torch.tensor(
        
        [[
            [[0,1,0],
             [0,0,0],
             [0,0,0]],
            [[0,1,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]]
        ],
         [
            [[0,1,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [0,0,0],
             [0,0,0]]
        ],
        ]
        
    ).float()
    
    scorer = DiceScore()
    dice = scorer(y_pred,y_gt)
    y_pred *=0
    y_gt = y_pred
    dice = scorer(y_pred,y_gt)
    print(dice)
    """