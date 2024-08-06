'''
@Author: ZZT
@Create Date: 2024/07/31
@Description: decoder，multimodal output，like HiVT
两部分：一个是轨迹回归的[F,N,H,4]，一个是模态概率分类的[F,N]，各自由一个MLP实现;

相较于unimodal的解码器，该解码器的主要改动在于将最后交叉注意力模块的输出分别送入上述的两个MLP中
'''


from typing import Optional,Tuple,Union
import torch
import torch.nn as nn
from torch import FloatTensor
import torch.nn.functional as F
from ..config import TDNASTConfig
from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock

class MultimodalDecoder(nn.Module):
    def __init__(self,
                 config:TDNASTConfig):
        super(MultimodalDecoder,self).__init__()
        self.config = config
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_blocks)])
        self.mlp = nn.Linear(config.embed_dim,config.channels)
        self.linear = nn.Linear(self.config.prediction_length,self.config.num_modes)
        self.cross_attn_output_dim = int(config.embed_dim / config.num_modes)
        self.futute_steps = config.prediction_length
        
        # 决定是否要返回分布的尺度参数
        self.uncertain = config.uncertain
        # 用于解码轨迹回归的MLP
        self.loc = nn.Sequential(
            nn.Linear(self.cross_attn_output_dim,self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.embed_dim, 2))
        if self.uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.cross_attn_output_dim, self.config.embed_dim),
                nn.LayerNorm(self.config.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.embed_dim,  2))
        # 用于解码模态概率分类的MLP
        self.pi = nn.Sequential(
            nn.Linear(self.cross_attn_output_dim, self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.embed_dim,1))
         
        
    def forward(self,
                hidden_states:FloatTensor,
                key_value_states:Tuple[FloatTensor,FloatTensor],
                attention_mask:Optional[FloatTensor]=None,
                return_attention:bool=False):
        dec_attentions = []
        for decoder_block in self.blocks:
            hidden_states, spatial_temporal_attn, cross_attn = decoder_block(
                hidden_states = hidden_states,
                key_value_states = key_value_states,
                attention_mask = attention_mask,
                return_attention = True)
            if return_attention:
                dec_attentions.append(spatial_temporal_attn)
        # 多模态输出，两部分：[F,N,H,4]轨迹回归（回归） 和 [F,N]模态概率（分类）
        '''
        目前的交叉注意力模块输出的hidden_states shape(batchsize,num_obj,pred_len,embed_dim)
        所以需要扩展维度啊，因为hidden里的四维是包括了batchsize的，所以要想办法扩展出一个F,只能从embed_dim里了
        '''
        multimodel_hidden_states = hidden_states.unsqueeze(1).view(self.config.batchsize,
                                                                   self.config.num_modes,
                                                                   self.config.num_objects,
                                                                   self.config.prediction_length,
                                                                   -1).contiguous()
        # 扩展后的shape （batchsize,F,N,H, -1),现在即需要把-1这维（其实等于embed_dim除以num_modes）映射到位置参数和尺度参数中去
        # pi = self.pi(hidden_states).squeeze(-1) # 预期shape [batchsize,N,F]
        pi = self.pi(multimodel_hidden_states)

        
        loc = self.loc(multimodel_hidden_states)
        if self.uncertain:
            scale = F.elu_(self.scale(multimodel_hidden_states),alpha=1.0) + 1.0
            scale = scale + self.config.min_scale   
            return torch.cat((loc,scale),dim=-1),pi
        else:
            return loc,pi
        
        # if return_attention:
        #     return hidden_states, dec_attentions[-1]
        # return hidden_states
    

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
        pos_encoded_hidden_states = (
            (hidden_states + positional_encoding).transpose(1,2).contiguous()
        )
        
        if key_value_states is not None:
            key_value_states = key_value_states.transpose(1,2).contiguous()
        
        # 计算时空注意力，和编码器中的时空注意力计算一样的
        decoder_output , temporal_attn = self.self_attention(
            pos_encoded_hidden_states,
            key_value_states=None,
            attention_mask=attention_mask,
            axis='time'
        )
        
        _, spatial_attn = self.self_attention(
            hidden_states,
            key_value_states=None,
            attention_mask=attention_mask,
            axis='space'
        )
        
        # 同样地计算联合的时空注意力
        spatial_temporal_attn = torch.matmul(spatial_attn,temporal_attn)
        decoder_output = torch.matmul(spatial_temporal_attn,decoder_output.transpose(1,2).contiguous())

        decoder_output, cross_attn = self.cross_attention(
            decoder_output.transpose(1,2).contiguous(),
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            axis='time'
        )
        
        decoder_output = decoder_output.transpose(1,2).contiguous()

        decoder_output = self.mlp(decoder_output)
        if return_attention:
            return decoder_output, spatial_temporal_attn,cross_attn
        return decoder_output
    
    