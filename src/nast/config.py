from dataclasses import dataclass

@dataclass
class TDNASTConfig:
    observation_length:int = 74 # 3s
    prediction_length:int = 125 # 5s
    num_objects:int = 8 # 在highd中同时考虑了自车周围8方位的车辆
    channels:int = 7 # 输入序列的特征维度
    
    encoder_blocks:int = 2 
    decoder_blocks:int = 2
    
    embed_dim:int = 72 # 最好是6的倍数
    encoder_attn_heads:int = 4
    decoder_attn_heads:int = 4
    encoder_attn_dropout:float = 0.1
    decoder_attn_dropout:float = 0.1
    
    encoder_ff_expansion:int = 4
    encoder_ff_dropout:float = 0.1
    decoder_ff_expansion:int = 4
    decoder_ff_dropout:float = 0.1
    

    uncertain:bool = True
    min_scale:float = 1.0
    
    batchsize:int = 128
    
    num_modes:int = 6

    