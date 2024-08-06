'''
@Author: ZZT
@Create Date: 2024/07/30
@Description: Rregression Loss 
use Laplace distribution to calculate the negative log-likelihood loss,the output through a mlp
                output shape (F,N,H,4)
                F is the number of mixture compoents,
                N is the number of agents in the scene (belike the num_obj),
                H is the number of predicted future time steps,
                4 is the 二维拉普拉斯分布的参数，包括位置参数mu和尺度参数b
'''

import torch
import torch.nn as nn 

class LaplaceNLLLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps # 这个eps的设置是为了数值稳定性的考虑？
        self.reduction = reduction
    
    def forward(self,
                pred:torch.Tensor,
                target:torch.Tensor) -> torch.Tensor:
        # print('- print the pred shape is : ',pred.shape)
        loc , scale = pred.chunk(2,dim=-1) # 沿张量pred最后一个维度拆分为两部分
        scale = scale.clone()
        with torch.no_grad():
            '''
            clamp_ 是pytorch里的一个张量方法，用于在原地（inplace）将张量中的每一个元素的值限定在指定范围内，参数可选为(min,max)
            '''
            scale.clamp_(min=self.eps)
        # print('- print the loc shape is : ',loc.shape)
        # print('- print the target shape is : ',target.shape)
        nll = torch.log(2*scale) + torch.abs(target - loc) / scale # 推导后确实是长这样
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}')
