# **说明**

## 各模块作用

### dataset

存储MNIST数据集

数据集由tensorflow下载，格式为npz文件，训练集特征/标签，测试集特征/标签在其中分别命名为'x_train'/'y_train'，'x_test'/'y_test'

### history

存储历史识别记录

文件结构如下：

```python
'''
./history/
    history_name1.pkl  # 历史记录pkl文件，存储数据类型为字典，其中史纪录默认采用pickle存储，为字典格式，一个元素内容为：
                           key=img_path, val=(cur_time, dgt_cnt)
                           img_path: 本次传入的图片文件路径
                           cur_time: 处理时间（在调用image_read_and_treat时获取），格式为
                           			 'YYYYmmdd_HHMMSS'（如'20240419_195458'）
                           dgt_cnt: 图片中包含的数字个数
    history_name2.pkl
    ...
    history_name1/
        cur_time1/ # 识别时间
            img_marked.jpg	# 标记后的图片
            img_npzs/	# 每个数字的神经网络输入
                1.npz
                ...
        cur_time2/
        ...
    history_name2/
    ...
'''
```

### models

以pkl格式存储训练完成的模型

### test_imgs

用于验证性能的测试图片

#### dataset_load.py

从文件中导入数据集，并转换成神经网络的输入

（*使用的Z-Score归一化是在和其他数据预处理方法（保持原始值不变、二值化）对比之后优选出来的*）

#### debug.py

使用test_imgs中的图片进行识别效果测试

#### hyperparameters_adjust.py

开发过程中用于**调参**

#### img_preprocess_and_cache.py

把实际场景的图片中所有的数字部分提取出来，转换成和MNIST数据集中样本相同的形式，再用与训练时相同的方法归一化，成为神经网络的输入；同时将标记出数字轮廓的图片和每个数字对应的神经网络输入缓存到history

#### main.py

包含了调用一个训练好的模型识别实际图片的全过程

#### network.py

network_layer和BP_Network类的实现（包含网络的训练，测试，持久化）

#### train_example.py

提供一个训练模型的代码示例



## 基本功能

1. 把单个数字的图片转换为28 * 28的灰度图像，再转换成神经网络输入，并缓存
2. 对含有多个数字的图片逐个提取，识别
3. 展示从图片中提取的数字，和标记出数字轮廓的原图



## 调用方法（如'效果展示.mp4')

1. 将目标图片的路径复制下来
2. 运行main.py，将路径粘贴至命令行中