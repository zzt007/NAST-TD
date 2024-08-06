'''
@Author: ZZT
@Create Date: 2024/07/28
@Description: model's encoder 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor
from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock
from ..config import TDNASTConfig

class Encoder(nn.Module):
    def __init__(self,config:TDNASTConfig):
        super().__init__()
        self.config = config
        self.channel_embed = nn.Linear(config.channels,config.embed_dim)
        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_blocks)])
        
    def forward(self,
                input_sequences:FloatTensor,
                return_attention:bool=True):
        # embedding for input
        hidden_states = self.channel_embed(input_sequences)
        hidden_states = functional.dropout(hidden_states,p=self.config.encoder_ff_dropout,training=self.training)
        hidden_states = self.layernorm(hidden_states)
        enc_self_attentions = []
        for encoder_block in self.blocks:
            hidden_states, attn_spt = encoder_block(
                hidden_states = hidden_states,
                return_attention = True
            )
            if return_attention:
                enc_self_attentions.append(attn_spt)
        if return_attention:
            return hidden_states, enc_self_attentions[-1]
        return hidden_states
    
    
class EncoderBlock(nn.Module):
    def __init__(self,config:TDNASTConfig):
        super(EncoderBlock,self).__init__()
        
        self.config = config
        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.observation_length,config.embed_dim),
            freeze=True
        )

        self.attention = MultiheadAttention(
            num_heads = config.encoder_attn_heads,
            embed_dim = config.embed_dim,
            attn_dropout = config.encoder_attn_dropout,
            ff_dropout = config.encoder_ff_dropout
        )
        
        self.mlp = FeedForwardBlock(config.embed_dim,
                                    config.encoder_ff_expansion,
                                    ff_dropout = config.encoder_ff_dropout)
    def forward(self,
                hidden_states:FloatTensor,
                key_value_states:Optional[FloatTensor]=None,
                attention_mask:Optional[FloatTensor]=None,
                return_attention:bool=False
                ) -> Union[FloatTensor,Tuple[FloatTensor,FloatTensor]]:
        # 对embed进行位置编码，用于时间注意力的计算（以每个时间步为间隔，所以是针对时间的，空间上不需要位置编码）
        positions = torch.arange(0,self.config.observation_length,dtype=torch.long).to(hidden_states.device)
        # positional_encoding shape (74,64) 即（obs_len,embed_dim）
        positional_encoding = self.pos_encode(positions)
        
        # 隐藏状态加上位置编码，并reshape
        hidden_states = hidden_states.transpose(1,2).contiguous() # 即shape从（64，74，8，64） -> （64，8，74，64）
        pos_encoded_hidden_states = (hidden_states + positional_encoding).transpose(1,2).contiguous()
        # 保存原来的hidden_states，作为空间注意力的输入表示，无需位置编码
        hidden_states = hidden_states.transpose(1,2).contiguous()

        if key_value_states:
            key_value_states = key_value_states.transpose(1,2).contiguous()
        
        # 计算时空注意力
        ## 计算时间的，是有输出的output的，即对应时间上的v
        encoder_output, temporal_attn = self.attention(
            pos_encoded_hidden_states,
            key_value_states = key_value_states,
            attention_mask = attention_mask,
            axis = 'time'
        )
        
        ## 计算空间的，不需要output，只需要q*k
        _, spatial_attn = self.attention(
            hidden_states,
            key_value_states = key_value_states,
            attention_mask = attention_mask,
            axis = 'space'
        )
        
        # 创建文中所述的'temporal influence map' ,并于空间注意力相乘，得到时空注意力
        spatial_temporal_attn = torch.matmul(spatial_attn,temporal_attn.transpose(1,2).contiguous())
        encoder_output = torch.matmul(
            spatial_temporal_attn.transpose(1,2).contiguous(),
            encoder_output.transpose(1,2).contiguous()
        )
        
        # 编码再经过一个mlp输出，达到d_model维度
        encoder_output = self.mlp(encoder_output)
        # print('- now finish the first block of encoder, encoder_output shape is :',encoder_output.shape)
        
        if return_attention:
            return encoder_output.transpose(1,2),spatial_temporal_attn
        
        return encoder_output.transpose(1,2)