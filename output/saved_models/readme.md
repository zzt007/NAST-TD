# 使用方法
'''
1、已按模块设计好不同的解码器可供选择，包括MLP解码器、自回归解码器AutoRegressive、非自回归解码器NAST
在model.py中选择不同的解码器，对应解开注释即可

2、当使用自回归解码器时，需要将观测序列和预测序列信息同时输入到模型中，因此需要在train和test部分，将pred()的输入进行修改

3、当使用自回归解码器时，注意注意力掩码矩阵的生成（与batchsize耦合了），当取batchsize为128时需要申请30GB显存，所以视自身情况进行修改即可

4、有对应的config文件，由于模型中的部分参数与其中的batchsize数值进行了耦合，因此更换batchsize时要在该config文件中修改

5、目前只能使用single-*文件，multi*的还没完成

6、用的时候记得自己修改数据来源目录
'''



# 数据来源
highd，将icus论文所用的拓展了维度num_obj=8
后来验证的时候再任选一维即可，经测试每一维都是一样的，这可能是因为原始数据也是复制得来的


# 模型介绍
需要注意的是，推理时间以batchsize设为1为准
## 0729_training_truncated
在训练到50个epochs时强制停止训练所得的模型，其实早在20左右就已经停止了loss下降，看了一下tensorboard图，基本上算是挺好的收敛；
参数：
-epochs 100
-lr 0.001
-batch_size 64

-tensorboard: events.out.tfevents.1722253104.DESKTOP-7FGIUOA.53221.0

## 0729
下班前重新炼的一版，明天来写测试用例
参数：
-epochs 50
-lr 0.001
-batch_size 128

-tensorboard: events.out.tfevents.1722258241.DESKTOP-7FGIUOA.71315.0
该版在第29个epoch处取得最终保存的模型

- result is :  [0.7204738, 1.336568, 1.8849611, 2.3803, 2.8564794, 1.7627576828002929, 0.11171875, 80.35523891448975] 
耗时在80-83之间波动

## 0802
添加了TD LOSS，成功跑通，但效果未知，等待完成
参数：
-epochs 50
-lr 0.001
-batch_size 128
但是config里的embed_dim 从原来的64改成了72，为了方便适应unimodal和multimodal之间的转换

训练到第14个epoch时，已经是最佳的模型了，test_loss甚至在六位小数下都是0

- result is :  [0.39631194, 0.82864153, 1.3476156, 1.9877043, 2.7943807, 1.4323484778404236, 0.1765625, 112.60967254638672]

对比0729未添加TD loss的模型，该模型在每一时刻的预测效果都要更好，尤其是在预测时域的前半段
但其推理耗时也增长了不少

- 一点想法：因为现在的TD loss是只用了一阶差分，如果能对一阶差分进行加权，突显后程的重要性？是否会效果更好

### 0802_nast_512bs
只将batchsize调成512，试一下


## 0802_mlpdecoder 
使用了TD Loss，同时使用mlp解码
参数：
-epochs 50
-lr 0.001
-batch_size 128

- result is :  [0.36976463, 0.71034324, 1.0842985, 1.5122371, 2.0413802, 1.1018519520759582, 0.41640625, 50.01654624938965]

这结果出乎意料了，简直是又快又好啊！

所以接下来需要：
1、和用MLP解码的模型对比精度 （x） 失败了，还不如这个
2、和用自回归解码的模型对比速度

## 0803_ardecoder
自回归解码的训练和测试流程已跑通，但是效果非常差，不知道是不是写错了。(特别是在生成掩码矩阵的时候，因为我和batchsize锚定了，所以还不能把batchsize设的太大，128时都需要申请30GB显存，我现在设为16)
现在重新训练一版使用自回归的模型再测试
-epochs 50
-lr 0.0001
-batch_size 16

- result is :  [0.42993376, 0.72176653, 0.9371334, 1.1402042, 1.4585242, 0.8865677111370619, 0.7267441860465116, 894.3926151408706]

以下是将batchsize设为1进行推理的
- result is :  [0.39192706, 0.6601631, 0.8338289, 0.959755, 1.1779975, 0.7895512893135378, 0.7267441860465116, 394.71077018005906]

这效果很好啊，只是解码比较慢
那再以batchsize 16训练一版NAST的

### 0803_nast
参数：
-epochs 50  （在第12个epoch就best了）
-lr 0.0001
-batch_size 16

- result is :  [0.44211853, 0.79698414, 1.1231991, 1.4632107, 1.8920856, 1.0920451311177986, 0.5, 10.379558385804643]

### 0803_nast_2
-epochs 30
-lr 0.0001
-batch_size 16
CLIP = 2 (原来是1)

- result is :  [0.3721882, 0.77785015, 1.2739027, 1.8980793, 2.692476, 1.3686659696490266, 0.20276162790697674, 20.35816048466882] 
前面效果不错，随预测时域增长，误差也快速上升。

CLIP = 3
- result is :  [0.9146187, 1.9147156, 3.0132844, 4.2364926, 5.6160173, 3.0526211677595625, 0.010174418604651164, 14.279933862907942]

不过在CLIP =3时，我只设置了20epochs，但却在19个epochs就best，不知道有没有到底

- 更换参数
-epochs 50
-lr 0.0001
-batch_size 8
-CLIP = 2 

- result is :  [0.84010655, 1.5770049, 2.2396681, 2.8294415, 3.368693, 2.090482009011646, 0.037063953488372096, 14.111904210822527]

- 更换参数
- lr = 0.001
- result is :  [2.208643, 4.3136263, 6.3304863, 8.246793, 10.054366, 6.020325078520664, 0.0, 10.993214540703352]

- 更换参数
- lr = 0.00001
- CLIP = 1
- result is :  [0.36174378, 0.6552114, 0.9581031, 1.3282796, 1.8371211, 0.9841748323551444, 0.5188953488372093, 13.647789178892623]
  尽管设置了50epochs，但在24个epoch时就best了

### 0803_nast_3
参数
-epochs 50
-lr 1e-5
-batch_size 16
-CLIP = 2

- result is :  [0.330027, 0.6164047, 0.9399347, 1.3581926, 1.9333085, 0.9967580645583397, 0.49055232558139533, 10.793647100759108]
其实前3s的效果还是很不错的，后程增长有点快；整体模型在40epoch左右达到best

### 0803_nast_4
参数
-epochs 70
-lr 1e-5
-batch_size 8
-CLIP = 1.8

- result is :  [0.49559966, 0.8972693, 1.2560991, 1.604499, 2.0086205, 1.1977230913417285, 0.42296511627906974, 10.597539502520894]
在32 epoch时就best了

### 0803_nast_5
参数
-epochs 70
-lr 1e-5
-batch_size 32
-CLIP = 1.8
- result is :  [0.5673193, 1.199486, 1.9251844, 2.7767205, 3.7913957, 1.9993036139843077, 0.038517441860465115, 11.018539583960244]

## 0804_nast_1
尝试改变一些别的，比如将Adam优化器替换为AdamW
参数
-epochs 50
-lr 1e-5
-batch_size 16
-CLIP = 1.8
- result is :  [1.2659466, 2.4051552, 3.4586642, 4.4410586, 5.3381968, 3.2603386585102525, 0.038517441860465115, 12.019637019135232]

### 0804_nast_2
替换优化器后，效果并不好，可能是其他的一些参数没有调好
参数
-epochs 50
-lr 1e-4
-batch_size 16
-CLIP = 1
- result is :  [1.2700157, 2.4980264, 3.865967, 5.285493, 6.7354302, 3.813133871832559, 0.010174418604651164, 10.860418164452842]

看了一下loss_train曲线，感觉50个的时候没怎么收敛，可能需要再加长一下epochs

### 0804_nast_3
-epochs 100
-lr 1e-4
-batch_size 16
-CLIP = 1
- result is :  [0.290968, 0.48781225, 0.6877774, 0.9626018, 1.392604, 0.7258952544179074, 0.7369186046511628, 13.856613358785939]

哟，可以喔，这版不错子。
- result is :  [0.23791605, 0.4029652, 0.5626643, 0.7626224, 1.0819677, 0.6099140430045271, 0.7369186046511628, 3.4692115908445316]