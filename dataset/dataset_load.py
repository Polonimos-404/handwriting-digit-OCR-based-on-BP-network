import numpy as np


def train_treat(filename: str = 'mnist.npz'):
    """
    数据集预处理
    格式要求：npz文件，训练集特征/标签，测试集特征/标签在存储时分别命名为'x_train'/'y_train'，'x_test'/'y_test'
    :param filename: 数据集文件名
    """
    with np.load(filename, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    # 将特征集每个数据扁平化为28 * 28 = 784维行向量，并归一
    x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype(np.float32)
    # 0.1307，0.3081分别为MNIST数据集灰度像素值压缩至(0, 1)区间后的均值和标准差
    x_train = (x_train / 255 - 0.1307) / 0.3081
    x_test = (x_test / 255 - 0.1307) / 0.3081
    # 为了方便计算误差，再单独生成一份使用one_hot编码的训练集标签
    y_train_one_hot = np.zeros((y_train.shape[0], 10))
    y_train_one_hot[[i for i in range(y_train.shape[0])], y_train] = 1
    print('Dataset preprocession completed.')
    return x_train, y_train, y_train_one_hot, x_test, y_test
