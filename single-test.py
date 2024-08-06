import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm

from src.nast.model import TDNAST
from src.nast.config import TDNASTConfig
from src.nast.losses.TD import TimeDependencyLoss

def feature_scaling(x_seq):
    x_min = x_seq.min() # 序列数据最小值
    x_max = x_seq.max() # 序列数据最大值
    # 如果最大等于最小，即整条序列都一样的值；否则，进行归一化处理
    if x_min == x_max:
        x_new = x_min * np.ones(shape=x_seq.shape)
    else:
        x_new = (2 * x_seq - (x_max + x_min)) / (x_max - x_min)
    return x_new, x_min, x_max

def de_feature_scaling(x_new, x_min, x_max):
    x_ori = np.ones(shape=(len(x_max), 200, 7)) # 根据目前的输入序列长度与特征
    for i in range(len(x_max)):
        for j in range(3):
            if x_min[i, j] == x_max[i, j]:
                x_ori[i, :, j] = x_min[i, j]
            else:
                x_ori[i, :, j] = (x_new[i, :, j] * (x_max[i, j] - x_min[i, j]) + x_max[i, j] + x_min[i, j]) / 2

    return x_ori

def data_diff(data):
    data_diff = np.diff(data)
    data_0 = data[0]
    return data_0, data_diff

def de_data_diff(data_0, data_diff):
    data = np.ones(shape=(len(data_diff), 200, 7))
    data[:, 0, :] = data_0
    for i in range(199):
        data[:, i + 1, :] = data[:, i, :] + data_diff[:, i, :]

    return data

def dataNormal(seq):
    seq_len = len(seq)
    seq_norm = np.zeros(shape=(seq_len, 199, 7))
    seq_norm_feature = np.zeros(shape=(seq_len, 3, 7))

    for i in range(seq_len):
        for j in range(7):
            seq_tmp = seq[i, :, j]  # initial seq
            seq_tmp_FS, seq_tmp_min, seq_tmp_max = feature_scaling(seq_tmp)  # feature scaling 对序列进行归一化，返回最大值和最小值，用于后续反归一化
            seq_tmp_0, seq_tmp_diff = data_diff(seq_tmp_FS)  # seq diff 序列数据差分处理，返回序列第一个值和后续元素关于第一个值的差值
            seq_norm[i, :, j] = seq_tmp_diff  # store norm data 返回数据差分后的值

            # store norm feature data，保存进行了数据处理的值，用于后续复原原始值，包括归一化所需的最大最小值、数据差分所需的起始值
            seq_norm_feature[i, 0, j] = seq_tmp_min
            seq_norm_feature[i, 1, j] = seq_tmp_max
            seq_norm_feature[i, 2, j] = seq_tmp_0

    return seq_norm, seq_norm_feature

def get_train_dataset(train_data, batch_size):
    x = train_data[:, :74, :]
    y = train_data[:, 74:, :]

    x_data = torch.from_numpy(x.copy())
    y_data = torch.from_numpy(y.copy())

    x_data = x_data.to(torch.float32)
    y_data = y_data.to(torch.float32)
    # 使用TensorDataset将观测序列和预测序列组成一条训练数据
    train_dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader

# 注意在生成验证和测试数据集时，有3个输入参数，其中传入了test_seq_NF,这是为了后续对数据进行复原处理，包括归一化和数据差分的，然后可以画图，所以后面也只在验证集生成的时候传入了这个参数
def get_test_dataset(test_data, test_seq_NF, batch_size):
    x_data = torch.from_numpy(test_data.copy())
    x_data = x_data.to(torch.float32)

    y_data = torch.from_numpy(test_seq_NF.copy())
    y_data = y_data.to(torch.float32)

    test_dataset = TensorDataset(x_data, y_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return test_loader

def LoadData():
    # train_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/train_data.npy')
    train_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_down/train_data.npy')

    
    # test_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/test_data.npy')
    test_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_down/test_data.npy')

    
    # valid_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/val_data.npy')
    valid_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_down/val_data.npy')

    return train_x, test_x, valid_x

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    
    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x,y))
        return loss
        
class RMSEsum(torch.nn.Module):
    def __init__(self):
        super(RMSEsum,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss(reduction="none")
        loss_sum = torch.sqrt(torch.sum(criterion(x, y), axis=-1))

        return loss_sum 



if __name__ == '__main__':
    '''
    由于训练时已经扩展维度为（batchsize, seq_len, num_obj, input_dim），但是是通过在对原始数据归一化后进行repeat得到的，
    所以在测试时，有两点需要注意：
    1、valid数据也需要相应地进行维度扩展，以适应模型的输入需要
    2、模型输出结果pred只选取num_obj中的一维进行计算指标（当然也可以尝试多次，虽然感觉上应该每一维结果都一样）
    '''
    result = [0,0,0,0,0,0,0,0] # [1s,2s,3s,4s,5sDE,RMSE,MR,Time]
    
    # 读取数据并且将val数据维度扩展
    seq_train, seq_test,seq_valid = LoadData()
    x_norm_train, x_norm_train_feature = dataNormal(seq_train)
    x_norm_test, x_norm_test_feature = dataNormal(seq_test)
    x_norm_valid, x_norm_valid_feature = dataNormal(seq_valid)
    
    # 只取前7个特征
    x_norm_train = x_norm_train[:, :, :7]
    x_norm_test = x_norm_test[:, :, :7]
    x_norm_valid = x_norm_valid[:, :, :7]
    
    # 维度扩展
    num_obj = 8
    x_norm_valid = np.repeat(np.expand_dims(x_norm_valid,axis=2),num_obj,axis=2)
    print('- now expand the valid data dimension is : ',x_norm_valid.shape)
    
    # dataloader
    batchsize = 1 # MLPdecoder时要改成当初训练时用的batchsize128，因为当时把它耦合了
    epochs = 1
    # 需要注意的是，这里的x_norm_valid_feature仍为三维的，所以到时候用的时候要先把预测结果从扩展的四维再降回三维
    valid_loader = get_test_dataset(x_norm_valid, x_norm_valid_feature, batch_size=batchsize)
    
    # 加载模型
    model_name = '/root/trajectory_prediction/TD-NAST/output/saved_models/TDNAST_test_0802_mlpdecoder.pt'
    model = torch.load(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义评价指标
    rmse_loss = RMSELoss()
    rmse_sum = RMSEsum()
    MR1 = 0
    MR2 = 0
    
    loss_tmp = []
    DE_1s_tmp = []
    DE_2s_tmp = []
    DE_3s_tmp = []
    DE_4s_tmp = []
    DE_5s_tmp = []
    inference_time_tmp = []
    
    for batch_idx,(x,x_NF) in enumerate(valid_loader):
        x = x.to(device)
        x_NF = x_NF.to(device)
        
        x = x.to(torch.float32)
        x_NF = x_NF.to(torch.float32)
        
        x_ori = x.clone()
        x_tmp = x[:,:74,:,:]
        y_tmp = x[:,74:,:,:]
        
        with torch.no_grad():
            start_time = time.time()
            pred = model(x_tmp,y_tmp)
            end_time = time.time()
            inference_time_tmp.append(end_time-start_time)
            
            pred = pred.to(device)
            pred = pred.transpose(1,2).contiguous()
            # print('- now print the pred shape is : ',pred.shape) # (batchsize, pred_len, num_obj, input_dim)
        # if use the mlp decoder , the pred shape is (batchsize,pred_len,num_obj,2), so the x at dimension 3 is set to :2
        x[:,74:,:,:2] = pred[:,:,:,:2]
        x_NF = x_NF.cpu().numpy()

        pred_seq_np_total = x.cpu().numpy()
        # 计算反归一化时，只取其中一维进行计算
        pred_seq_np = pred_seq_np_total[:,:,0,:]
        pred_seq_dediff = de_data_diff(x_NF[:,2,:], pred_seq_np)
        pred_seq_ori = de_feature_scaling(pred_seq_dediff,x_NF[:,0,:],x_NF[:,1,:])
        
        # 对x也是只取1维
        x_ori = x_ori[:,:,0,:]
        x_ori_np = x_ori.cpu().numpy()
        x_ori_dediff = de_data_diff(x_NF[:,2,:], x_ori_np)
        x_ori_oo = de_feature_scaling(x_ori_dediff,x_NF[:,0,:],x_NF[:,1,:])

        pred_seq_ori_torch = torch.from_numpy(pred_seq_ori)
        x_data_oo_torch = torch.from_numpy(x_ori_oo)

        pred_seq_ori_torch = pred_seq_ori_torch.to(torch.float32)
        x_data_oo_torch = x_data_oo_torch.to(torch.float32)

        loss_1s = rmse_loss(x_data_oo_torch[:, 99,  :2], pred_seq_ori_torch[:, 99,  :2])
        loss_2s = rmse_loss(x_data_oo_torch[:, 124, :2], pred_seq_ori_torch[:, 124, :2])
        loss_3s = rmse_loss(x_data_oo_torch[:, 149, :2], pred_seq_ori_torch[:, 149, :2])
        loss_4s = rmse_loss(x_data_oo_torch[:, 174, :2], pred_seq_ori_torch[:, 174, :2])
        loss_5s = rmse_loss(x_data_oo_torch[:, 199, :2], pred_seq_ori_torch[:, 199, :2])

        loss = rmse_loss(x_data_oo_torch[:, 75:, :2], pred_seq_ori_torch[:, 75:, :2])

        DE_1s_tmp.append(loss_1s)
        DE_2s_tmp.append(loss_2s)
        DE_3s_tmp.append(loss_3s)
        DE_4s_tmp.append(loss_4s)
        DE_5s_tmp.append(loss_5s)
        loss_tmp.append(loss.item())
        
        rmse_5s_ = rmse_sum(x_data_oo_torch[:, 199, :2], pred_seq_ori_torch[:, 199, :2])
        rmse_5s = np.array(rmse_5s_)
        
        MR1 = MR1 + np.sum(rmse_5s>=2)
        MR2 = MR2 + np.sum(rmse_5s<=2)

        loss_x_1s = rmse_loss(x_data_oo_torch[:, 99, 0], pred_seq_ori_torch[:, 99, 0])
        loss_x_2s = rmse_loss(x_data_oo_torch[:, 124, 0], pred_seq_ori_torch[:, 124, 0])
        loss_x_3s = rmse_loss(x_data_oo_torch[:, 149, 0], pred_seq_ori_torch[:, 149, 0])
        loss_x_4s = rmse_loss(x_data_oo_torch[:, 174, 0], pred_seq_ori_torch[:, 174, 0])
        loss_x_5s = rmse_loss(x_data_oo_torch[:, 199, 0], pred_seq_ori_torch[:, 199, 0])
        
        loss_y_1s = rmse_loss(x_data_oo_torch[:, 99, 1], pred_seq_ori_torch[:, 99, 1])
        loss_y_2s = rmse_loss(x_data_oo_torch[:, 124, 1], pred_seq_ori_torch[:, 124, 1])
        loss_y_3s = rmse_loss(x_data_oo_torch[:, 149, 1], pred_seq_ori_torch[:, 149, 1])
        loss_y_4s = rmse_loss(x_data_oo_torch[:, 174, 1], pred_seq_ori_torch[:, 174, 1])
        loss_y_5s = rmse_loss(x_data_oo_torch[:, 199, 1], pred_seq_ori_torch[:, 199, 1])

        
            
    loss_1s_tmp_np = np.array(DE_1s_tmp)
    loss_2s_tmp_np = np.array(DE_2s_tmp)
    loss_3s_tmp_np = np.array(DE_3s_tmp)
    loss_4s_tmp_np = np.array(DE_4s_tmp)
    loss_5s_tmp_np = np.array(DE_5s_tmp)
    inference_time_tmp = np.array(inference_time_tmp)
    
    loss_1s_mean = loss_1s_tmp_np.mean()
    loss_2s_mean = loss_2s_tmp_np.mean()
    loss_3s_mean = loss_3s_tmp_np.mean()
    loss_4s_mean = loss_4s_tmp_np.mean()
    loss_5s_mean = loss_5s_tmp_np.mean()

    inference_time_mean = np.mean(inference_time_tmp*1000)
    loss_tmp_np = np.array(loss_tmp)
    loss_mean = loss_tmp_np.mean()
    result[0] = loss_1s_mean
    result[1] = loss_2s_mean
    result[2] = loss_3s_mean
    result[3] = loss_4s_mean
    result[4] = loss_5s_mean
    result[5] = loss_mean
    result[6] = MR2/(MR1+MR2)       
    result[7] = inference_time_mean # ms             
    print('- result is : ',result)