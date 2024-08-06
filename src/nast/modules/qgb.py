'''
@Author: ZZT
@Create Date: 2024/07/28
@Description: QueryGenerationBlock, generate the decoder queries in one step,queries lenght same with prediction length
'''

'''
Given encoder output hidden states `encoder_ouptus` of shape `(b, c, th, e)`, produce a 
history score tensor of shape `(b th c)` representing the temporal influence of 
the input sequence (hidden states) on the queries. Use a positional 
embedding table to produce a positonal encoding of shape `(tf, 1)` which 
will represent spatial influence on the queries. Finally, synthesize the 
temporal and spatial influences via a `matmul` operation, resulting in joint 
influence `I` and generate queries of shape `(b c tf, e)` by performing
`torch.matmul(I.mT, encoder_outputs)`.
'''
import torch
import torch.nn as nn
from torch import FloatTensor
from .attention import positional_encoding_table
from ..config import TDNASTConfig

class QueryGenerationBlock(nn.Module):
    def __init__(self,config:TDNASTConfig,bias:bool=True):
        super(QueryGenerationBlock,self).__init__()
        self.config = config
        self.pos_embed = nn.Embedding.from_pretrained(
            positional_encoding_table(config.prediction_length,config.embed_dim),
            freeze=True
        )
        # 生成history score
        self.history_proj = nn.Linear(config.embed_dim, 1, bias=bias)
        # 生成位置编码得分
        self.pos_proj = nn.Linear(config.embed_dim,1,bias=bias)
        
    def forward(self,
                encoder_outputs:FloatTensor,
                return_encoder_outputs:bool=True):
        # encoder_output shape (batch_size, seq_len, num_obj, embed_dim)
        history_scores = self.history_proj(encoder_outputs)
        history_scores = torch.relu(history_scores)

        positions = torch.arange(0,self.config.prediction_length,dtype=torch.long).to(encoder_outputs.device)
        position_encoding = self.pos_embed(positions)
        position_scores = self.pos_proj(position_encoding).transpose(0,1).contiguous()
        position_scores = torch.relu(position_scores)
        
        weights = torch.matmul(history_scores,position_scores)
        # 改变shape排列，此时的weights shape从（64，74，125，8）变为（64，8，125，74）
        weights = weights.permute(0,2,3,1)
        encoder_outputs = encoder_outputs.transpose(1,2).contiguous()
        # 怎么是和编码器输出相乘，不是和观测序列的embed相乘么？
        queries = torch.matmul(weights,encoder_outputs) 
        if return_encoder_outputs:
            # 返回queries和encoder_output作为cross_attention模块的输入
            return queries, encoder_outputs
        # 预期得到queries的shape为(batchsizen num_obj, pred_len,embed_dim)
        return queries
        
