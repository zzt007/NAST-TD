'''
@Author: ZZT
@Create Date: 2024/08/02
@Description: use a mlp to decode, which input is the encoder_output and obs_seq_embedding 
encoder_output shape (batchsize,num_obj,obs_len,embed_dim)
obs_seq_embedding shape (batchsize,num_obj,obs_len,input_dim)
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from ..config import TDNASTConfig

class MLPDecoder(nn.Module):
    def __init__(self,config):
        super(MLPDecoder,self).__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        self.batchsize = self.config.batchsize
        self.obs_len = self.config.observation_length
        self.pred_len = self.config.prediction_length
        self.input_dim = self.embed_dim * self.obs_len
        self.output_dim = self.pred_len * 2
        self.fc1 = nn.Linear(self.input_dim,self.input_dim * 2)
        self.fc2 = nn.Linear(self.input_dim * 2,self.output_dim)
        self.layernorm = nn.LayerNorm(self.input_dim * 2)
        
        self.fc3 = nn.Linear(self.config.channels,self.config.embed_dim)
        
    def forward(self,
                encoder_output,
                obs_seq_embedding):
        '''
        aggregate the encoder_output and obs_seq_embedding,like the residual
        - encoder_output shape (batchsize,num_obj,obs_len,embed_dim)
        - obs_seq_embedding shape (batchsize,obs_len,num_obj,input_dim=7),还没经过位置编码的，而且num_obj也没换位置
        '''
        obs_seq_embedding = self.fc3(obs_seq_embedding) # (batchsize,num_obj,obs_len,embed_dim)
        obs_seq_embedding = obs_seq_embedding.permute(0,2,1,3)
        aggr = encoder_output + obs_seq_embedding
        
        loc = aggr.view(-1,self.input_dim) # (batchsize*num_obj,obs_len*embed_dim)
        loc = self.fc1(loc)
        loc = self.layernorm(loc)
        loc = torch.relu(loc)
        loc = self.fc2(loc)
        
        loc = loc.view(self.batchsize,
                       -1,
                       self.pred_len,
                       2)
        
        return loc
        
        