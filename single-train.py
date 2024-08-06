'''
@Author: ZZT
@Create Date: 2024/07/28
@Description: 
'''
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os 

from src.nast.model import TDNAST
from src.nast.config import TDNASTConfig
from src.nast.losses.TD import TimeDependencyLoss

# highd数据预处理
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
    x_ori = np.ones(shape=(len(x_max), 200, 19)) # 根据目前的输入序列长度与特征
    for i in range(len(x_max)):
        for j in range(3):
            if x_min[i, j] == x_max[i, j]:
                x_ori[i, :, j] = x_min[i, j]
            else:
                x_ori[i, :,j] = (x_new[i, :, j] * (x_max[i, j] - x_min[i, j]) + x_max[i, j] + x_min[i, j]) / 2

    return x_ori

def data_diff(data):
    data_diff = np.diff(data)
    data_0 = data[0]
    return data_0, data_diff

def de_data_diff(data_0, data_diff):
    data = np.ones(shape=(len(data_diff), 200, 19))
    data[:, 0, :] = data_0
    for i in range(199):
        data[:, i + 1, :] = data[:, i,:] + data_diff[:, i, :]

    return data

def dataNormal(seq):
    seq_len = len(seq)
    seq_norm = np.zeros(shape=(seq_len, 199, 19))
    seq_norm_feature = np.zeros(shape=(seq_len, 3, 19))

    for i in range(seq_len):
        for j in range(7):
            seq_tmp = seq[i, : ,j]  # initial seq
            seq_tmp_FS, seq_tmp_min, seq_tmp_max = feature_scaling(seq_tmp)  # feature scaling 对序列进行归一化，返回最大值和最小值，用于后续反归一化
            seq_tmp_0, seq_tmp_diff = data_diff(seq_tmp_FS)  # seq diff 序列数据差分处理，返回序列第一个值和后续元素关于第一个值的差值
            seq_norm[i, :, j] = seq_tmp_diff  # store norm data 返回数据差分后的值

            # store norm feature data，保存进行了数据处理的值，用于后续复原原始值，包括归一化所需的最大最小值、数据差分所需的起始值
            seq_norm_feature[i, 0,j] = seq_tmp_min
            seq_norm_feature[i, 1,j] = seq_tmp_max
            seq_norm_feature[i, 2,j] = seq_tmp_0

    return seq_norm, seq_norm_feature

def get_train_dataset(train_data, batch_size):
    x = train_data[:, :74, :,:]
    y = train_data[:, 74:, :,:]

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
    train_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/train_data.npy')
    # train_y = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data/train_labels.npy')
    
    test_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/test_data.npy')
    # test_y = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data/test_labels.npy')
    
    valid_x = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data_slide_down/val_data.npy')
    # valid_y = np.load(file='/root/trajectory_prediction/icus_paper/highD_data/final_merge_data/val_labels.npy')
    # return train_x, train_y, test_x, test_y, valid_x, valid_y
    return train_x , test_x, valid_x


if __name__ == '__main__':
    seq_train, seq_test, seq_valid = LoadData()
    
    x_norm_train, x_norm_train_feature = dataNormal(seq_train)
    x_norm_test, x_norm_test_feature = dataNormal(seq_test)
    x_norm_valid, x_norm_valid_feature = dataNormal(seq_valid)
    
    # 只取前7个特征
    x_norm_train = x_norm_train[:, :, :7]
    x_norm_test = x_norm_test[:, :, :7]
    x_norm_valid = x_norm_valid[:, :, :7]
    
    
    # 在预处理后再进行扩展维度,因为是通过repeat来扩展的，最后要进行反归一化时，只需选取任一复制的一维即可
    num_obj = 8
    x_norm_train = np.repeat(np.expand_dims(x_norm_train, axis=2),num_obj,axis=2)
    x_norm_test = np.repeat(np.expand_dims(x_norm_test, axis=2),num_obj,axis=2)
    x_norm_valid = np.repeat(np.expand_dims(x_norm_valid, axis=2),num_obj,axis=2)
    print('-now expand the dimension of : ',x_norm_train.shape)
    
    # 训练相关参数
    batch_size = 16
    epochs = 100
    learning_rate = 0.0001
    CLIP = 2
    
    # 加载数据集
    train_loader = get_train_dataset(x_norm_train, batch_size)
    test_loader = get_train_dataset(x_norm_test, batch_size)
    valid_loader = get_test_dataset(x_norm_valid, x_norm_valid_feature, batch_size)
    
    # 实例化模型
    model = TDNAST(TDNASTConfig) # 已按config配置
    criterion = nn.SmoothL1Loss()
    td_criterion = TimeDependencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # tensorboard
    log_dir = '/root/trajectory_prediction/TD-NAST/runs/train'
    writer = SummaryWriter(log_dir)
    
    # 模型保存
    best_test_loss = float('inf')
    model_dir = '/root/trajectory_prediction/TD-NAST/output/saved_models/'
    model_name = 'TDNAST_test_0804_nast_3.pt'
    saved_model_path = model_dir + model_name
    
    # 定义训练过程
    def train(model,train_loader,optimizer,criterion,td_criterion,clip,device):
        model.train()
        train_loss = 0
        for i,(x,y) in enumerate(train_loader):
            # x shape (64,74,8,7) ,y shape(64,125,8,7)
            x = x.to(device)
            y = y.to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)

            # 如果是使用自回归解码，则还需要输入y并对y进行调换（64,8,125,7)，这一步就在model里操作吧，这里操作会影响下面的求损失
            pred = model(x)
            # pred = model(x,y)
            pred = pred.to(device)
            # print('- print the pred shape is :',pred.shape)
            pred = pred.transpose(1,2).contiguous()
            original_loss = criterion(pred[:,:,:,:2],y[:,:,:,:2])
            loss = td_criterion(pred,y,original_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    # 定义eval过程
    def evaluate(model,test_loader,criterion,td_criterion,device):
        model.eval()
        test_loss = 0
        for i,(x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)

            with torch.no_grad():
                # 当采用自回归解码时，还需要输入y
                pred = model(x)
                # pred = model(x,y)
                pred = pred.to(device)
                pred = pred.transpose(1,2).contiguous()
                original_loss = criterion(pred[:,:,:,:2],y[:,:,:,:2])
                loss = td_criterion(pred,y,original_loss)
                test_loss += loss.item()
        return test_loss / len(test_loader)
    

    # 开始训练
    for epoch in tqdm(range(epochs)):
        train_loss = train(model,train_loader,optimizer,criterion,td_criterion,CLIP,device)
        test_loss = evaluate(model,test_loader,criterion,td_criterion,device)
        print(F'Epoch: {epoch+1:02}')
        print(F'\tTrain Loss: {train_loss:.6f}')
        print(F'\t Test Loss: {test_loss:.6f}')
        
        writer.add_scalar('Loss/train',train_loss,epoch)
        writer.add_scalar('Loss/test',test_loss,epoch)
        writer.flush()

        if test_loss < best_test_loss:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model, saved_model_path)
            print('- finish training , the model has been saved !')
            best_test_loss = test_loss
            
    writer.close()
    