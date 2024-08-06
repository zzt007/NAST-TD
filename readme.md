# TD-NAST(Time Dependency Non-Autoregressive Spatial-Temporal vehilce trajectory prediction)
 
## 对highd数据集进行处理
输入的sequence shape为(batchsize,seq_len,num_obj,input_dim)
但是原来是没有考虑num_obj的，所以要在原有的数据基础上扩展该维度，目前考虑num_obj = 8
