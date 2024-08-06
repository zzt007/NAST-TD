'''
@Author: ZZT
@Create Date: 2024/07/28
@Description: attention block,includes multiheads attention and self scale-dot product attention 
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as functional
import numpy as np
from typing import Optional, Tuple, Union
from torch import FloatTensor

def positional_encoding_table(sequence_length:Union[int,FloatTensor],
                              embed_dim:int):
    # 正弦位置编码
    if isinstance(sequence_length,int):
        positions = list(range(sequence_length))
        
    def cal_angle(position,hid_idx):
        return position / np.power(1000.0, 2*(hid_idx // 2) / embed_dim)
    
    def get_posi_angel_vec(position):
        return [cal_angle(position,hid_j) for hid_j in range(embed_dim)]
    
    sinusoid_table = np.array([get_posi_angel_vec(pos_i) for pos_i in positions])
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])
    
    return torch.FloatTensor(sinusoid_table)

class ScaleDotProductAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 dropout:float = 0.0):
        super(ScaleDotProductAttention,self).__init__()
        self.scale = embed_dim**-0.5
        self.dropout = dropout
        
    def forward(self,
                queries:FloatTensor,
                keys:FloatTensor,
                values:FloatTensor,
                mask:Optional[FloatTensor]=None) -> Tuple[FloatTensor,FloatTensor]:
        # 为什么keys是2，3维交换？因为此时的keys/values/queries的shape为(-1,num_heads,obs_len,embed_dim/num_heads)
        attn = torch.matmul(queries*self.scale,keys.transpose(2,3))
        # 240802- 暂时将此处屏蔽，直接让0处还是0
        # if mask is not None:
        #     # 将mask从cpu转到gpu
            
        #     # 使用masked_fill函数将掩码中值为0的位置填充为-1e9（一个非常小的负数），这相当于在计算注意力权重时将这些位置设为无穷小，从而忽略它们的影响。
        #     attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = functional.softmax(attn,dim=-1)
        attn = functional.dropout(attn,p=self.dropout,training=self.training)

        out = torch.matmul(attn,values)
        return out,attn
    
class MultiheadAttention(nn.Module):
    '''
    需要注意的是，这里要区分'time' or 'space' axis，即是沿着哪个维度进行计算注意力
    当计算时间注意力时，把space当成是batch维度处理；
    当计算空间注意力时，把time当成是batch维度处理；
    '''
    def __init__(self,
                 num_heads:int,
                 embed_dim:int,
                 attn_dropout:float=0.0,
                 ff_dropout:float=0.0,
                 bias:bool=True):
        super(MultiheadAttention,self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        
        self.query_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.key_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.value_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.attn_proj = ScaleDotProductAttention(embed_dim,dropout=attn_dropout)

        self.fc = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self,
                hidden_states:FloatTensor,
                key_value_states:Optional[FloatTensor]=None,
                attention_mask:Optional[FloatTensor]=None,
                axis:str='time') -> Tuple[FloatTensor,FloatTensor]:
        batch_size,seq_len,num_obj,embed_dim = hidden_states.size()
        is_cross_attention = key_value_states is not None
        residual = hidden_states
        
        if axis == 'space':
            hidden_states = hidden_states.view(-1,num_obj,embed_dim)
        else:
            hidden_states = (
                hidden_states.transpose(1,2).contiguous().view(-1,seq_len,embed_dim)
            )
        queries: FloatTensor = self.query_proj(hidden_states)
        
        if is_cross_attention:
            # 这种获取序列长度的办法太容易出错了，而且kv也就在交叉注意力计算时才用到,这里要求传入的encout的shape为（batchsize，seq_len,num_obj,embed_dim)
            kv_len = key_value_states.size(1)
            # 直接赋值
            # kv_len = 74
            if axis == 'space':
                key_value_states = key_value_states.view(-1,num_obj,embed_dim)
            else:
                key_value_states = (
                    key_value_states.transpose(1,2).contiguous().view(-1,kv_len,embed_dim)
                )
            keys: FloatTensor = self.key_proj(key_value_states)
            values: FloatTensor = self.value_proj(key_value_states)
        else:
            keys: FloatTensor = self.key_proj(hidden_states)
            values: FloatTensor = self.value_proj(hidden_states)

        # 将注意力分到每个头上
        if axis == 'time':
            # 其实shape还是(bs,num_heads,seq_len,head_dim)，因为把原来的embed_dim 分成了head_dim x num_heads
            queries = queries.view(-1,self.num_heads,seq_len,self.head_dim)
            if is_cross_attention:
                keys = keys.view(-1,self.num_heads,kv_len,self.head_dim)
                values = values.view(-1,self.num_heads,kv_len,self.head_dim)
            else:
                keys = keys.view(-1,self.num_heads,seq_len,self.head_dim)   
                values = values.view(-1,self.num_heads,seq_len,self.head_dim)
        else:
            queries = queries.view(-1,self.num_heads,num_obj,self.head_dim)
            keys = keys.view(-1,self.num_heads,num_obj,self.head_dim)
            values = values.view(-1,self.num_heads,num_obj,self.head_dim)
        
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1)
        
        out, attn = self.attn_proj(queries,keys,values,mask=attention_mask)
        
        # reshape, average attention across heads
        out = out.view(batch_size,seq_len,num_obj,embed_dim)
        if axis == 'space':
            attn = attn.view(batch_size,seq_len,self.num_heads,num_obj,num_obj)
        else:
            attn = attn.view(batch_size,num_obj,self.num_heads,kv_len if is_cross_attention else seq_len, seq_len)
        
        attn = torch.mean(attn,dim=2)
        
        # fc + residual
        out = self.fc(out)
        out = functional.dropout(out,p=self.ff_dropout,training=self.training)
        out += residual
        
        # layernorm
        out = self.layernorm(out)

        return out,attn