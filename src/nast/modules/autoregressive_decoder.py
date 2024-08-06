'''
@Author: ZZT
@Create Date: 2024/07/28
@Description: decoder，for unimodal
'''

from typing import Optional,Tuple,Union
import torch
import torch.nn as nn
from torch import FloatTensor

from ..config import TDNASTConfig
from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock

class AutoRegressiveDecoder(nn.Module):
    def __init__(self,
                 config:TDNASTConfig):
        super(AutoRegressiveDecoder,self).__init__()
        self.config = config
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_blocks)])
        self.mlp = nn.Linear(config.embed_dim,config.channels)
        self.fc = nn.Linear(config.channels,config.embed_dim) # 仅用于将预测序列的维度转成高维表示
    def forward(self,
                hidden_states:FloatTensor,
                key_value_states:Tuple[FloatTensor,FloatTensor],
                attention_mask:Optional[FloatTensor]=None,
                return_attention:bool=False):
        dec_attentions = []
        
        # (batchsize,num_obj,pred_len,embed_dim)
        hidden_states = self.fc(hidden_states)
        
        for decoder_block in self.blocks:
            hidden_states, spatial_temporal_attn, cross_attn = decoder_block(
                hidden_states = hidden_states,
                key_value_states = key_value_states,
                attention_mask = attention_mask,
                return_attention = True)
            if return_attention:
                dec_attentions.append(spatial_temporal_attn)
        # 输出原始的channels维度
        hidden_states = self.mlp(hidden_states)
        
        if return_attention:
            return hidden_states, dec_attentions[-1]
        return hidden_states
    

class DecoderBlock(nn.Module):
    def __init__(self,config:TDNASTConfig):
        super(DecoderBlock,self).__init__()
        self.config = config
        
        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.prediction_length,config.embed_dim),
            freeze=True
        )
        self.self_attention = MultiheadAttention(
            num_heads = config.decoder_attn_heads,
            embed_dim = config.embed_dim,
            attn_dropout = config.decoder_attn_dropout,
            ff_dropout = config.decoder_ff_dropout
        )
        self.cross_attention = MultiheadAttention(
            num_heads = config.decoder_attn_heads,
            embed_dim = config.embed_dim,
            attn_dropout = config.decoder_attn_dropout,
            ff_dropout = config.decoder_ff_dropout
        )
        self.mlp = FeedForwardBlock(
            config.embed_dim,
            config.decoder_ff_expansion,
            ff_dropout = config.decoder_ff_dropout
        )
    
    def forward(self,
                hidden_states:FloatTensor,
                key_value_states:FloatTensor,
                attention_mask:Optional[FloatTensor]=None,
                return_attention:bool=False) -> Union[FloatTensor,Tuple[FloatTensor,FloatTensor,FloatTensor]]:
        # 对输入进行位置编码，供计算时间注意力时使用
        positions = torch.arange(0,self.config.prediction_length,dtype=torch.long).to(hidden_states.device)
        positional_encoding = self.pos_encode(positions)
        # pos_encoded_hidden_states即对预测序列的embed进行了位置编码，此时shape (bs,pred_len,num_obj,embed_dim)
        # print('- print the decoder forward''s hidden_states shape:',hidden_states.shape)
        pos_encoded_hidden_states = (
            (hidden_states + positional_encoding).transpose(1,2).contiguous()
        )
        
        # if key_value_states is not None:
        #     key_value_states = key_value_states.transpose(1,2).contiguous()
        
        # 计算时空注意力，和编码器中的时空注意力计算一样的
        decoder_output , temporal_attn = self.self_attention(
            pos_encoded_hidden_states,
            key_value_states=None,
            attention_mask=attention_mask,
            axis='time'
        )
        
        _, spatial_attn = self.self_attention(
            hidden_states.transpose(1,2).contiguous(),
            key_value_states=None,
            attention_mask=attention_mask,
            axis='space'
        )
        
        # 同样地计算联合的时空注意力,原来输出的是s_a (bs,pred_len,num_obj,num_obj), t_a (bs,num_obj,pred_len,pred_len)
        # 所以要转置一下，按论文里是转置temporal attn 变成所谓的temporal influence map
        temporal_attn = temporal_attn.transpose(1,2).contiguous() # (bs,pred_len,num_obj,pred_len)
        spatial_temporal_attn = torch.matmul(spatial_attn,temporal_attn).transpose(1,2).contiguous() # 与论文中一致(bs,num_obj,pred_len,pred_len)
        decoder_output = torch.matmul(spatial_temporal_attn,decoder_output.transpose(1,2).contiguous()) # trans后的decoder_outpout shape(bs,num_obj,pred_len,embed_dim)
        
        decoder_output, cross_attn = self.cross_attention(
            decoder_output.transpose(1,2).contiguous(),
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            axis='time'
        )
        
        decoder_output = decoder_output.transpose(1,2).contiguous() 
        # print('- now print the decoder_output shape:' ,decoder_output.shape)

        decoder_output = self.mlp(decoder_output)
        if return_attention:
            return decoder_output, spatial_temporal_attn,cross_attn
        return decoder_output