'''
@Author: ZZT
@Create Date: 2024/07/30
@Description: Classification Loss 
use soft-target-crossentropy loss to train the model
                output shape (F,N)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, reduction:str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self,
                pred:torch.Tensor,
                target:torch.Tensor) -> torch.Tensor:
        # print('- print the type of pred: ',type(pred))
        log_softmax = F.log_softmax(pred,dim=-1)
        cross_entropy = torch.sum(-target * log_softmax,dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
           