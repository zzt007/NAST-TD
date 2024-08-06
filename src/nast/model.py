import torch.nn as nn
from torch import FloatTensor
import sys
# print(sys.path)
from .config import TDNASTConfig

# 通过这里来改变是否为多模态
from .modules.decoder import Decoder
# from .modules.multimodal_decoder import MultimodalDecoder

# 是否使用MLP进行解码
from .modules.mlp_decoder import MLPDecoder

# 是否使用AutoRegressiveDecoder进行解码
from .modules.autoregressive_decoder import AutoRegressiveDecoder

from .modules.encoder import Encoder
from .modules.qgb import QueryGenerationBlock



class TDNAST(nn.Module):
    def __init__(self, config: TDNASTConfig):
        super(TDNAST, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.qgb = QueryGenerationBlock(config)
        
        self.decoder = AutoRegressiveDecoder(config)
        # self.decoder = MLPDecoder(config)
        # self.decoder = Decoder(config)
        # self.decoder = MultimodalDecoder(config)

    def forward(self, input_sequences: FloatTensor) -> FloatTensor:
        encout = self.encoder(input_sequences, return_attention=False)
        decoder_queries, encout = self.qgb(encout)
        # use mlp decoder 
        # loc = self.decoder(encout, input_sequences)
        # return loc
        
        # use unimodal decoder 
        # hidden_states = self.decoder(decoder_queries, encout)
        # return hidden_states
        
        # use multimodal decoder
        # loc,pi = self.decoder(decoder_queries, encout)
        # return loc,pi




    # if use autoregressive decoder , note that we should cancel the QGB and change the model's input 
    def forward(self,input_sequences: FloatTensor,pred_sequences:FloatTensor) -> FloatTensor:
        encout = self.encoder(input_sequences, return_attention=False)
        # 将pred_sequences从shape为(batchsize,pred_len,num_obj,input_channels) -> (batchsize,num_obj,pred_len,input_channels)
        pred_sequences = pred_sequences.transpose(1,2).contiguous()
        hidden_states = self.decoder(pred_sequences,encout)
        
        return hidden_states