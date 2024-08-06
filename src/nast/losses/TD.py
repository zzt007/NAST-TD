'''
@Author: ZZT
@Create Date: 2024/08/02
@Description: Time Dependency Loss 
calulate the time dependency loss for the given prediction and target,
which is divided into two parts: fine-grained and coarse-grained.
'''
import torch
import torch.nn as nn 

class TimeDependencyLoss(nn.Module):
    def __init__(self,
                 ):
        super(TimeDependencyLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        
    def forward(self,
                pred,
                groundtruth,
                original_loss):
        '''
        1、pred的一阶差分 以及 gt的一阶差分，但差分的shape是要比预测序列长度小1的
        2、计算rho
        3、粗细粒度loss合并，结合原始loss
        '''
        pred_current_steps = pred[:,:,1:,:2]
        pred_previous_steps = pred[:,:,:-1,:2]
        
        pred_diff = pred_current_steps - pred_previous_steps
        
        # print('- now print the pred_diff shape:', pred_diff.shape)
        
        gt_current_steps = groundtruth[:,:,1:,:2]        
        gt_previous_steps = groundtruth[:,:,:-1,:2]
        gt_diff = gt_current_steps - gt_previous_steps
        
        # distance metric such as MSE or MAE or SmoothL1
        fine_grained_loss = self.mse(pred_diff, gt_diff)
        
        # calulate the rho, which means the coarse-grained
        sign_pred_diff = torch.sign(pred_diff)
        sign_gt_diff = torch.sign(gt_diff)
        
        # 使用pytorch的ne方法（not equal）比较两个序列
        comparison = sign_pred_diff.ne(sign_gt_diff)
        # 将比较返回的bool类型转为int
        coarse_grained = comparison.int()
        # 计算结果中为1的个数占比
        num_ones = torch.sum(coarse_grained)
        total_elements = coarse_grained.numel()
        rho = num_ones / total_elements
        
        # combine the fine-grained and coarse-grained loss
        loss = rho * original_loss + (1 - rho) * fine_grained_loss
        
        return loss    
    