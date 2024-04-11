import numpy as np


def treat():
    with np.load('source/data/mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    # 为了方便计算误差，另单独生成一份one_hot编码的训练集标签
    y_train_one_hot = np.zeros((y_train.shape[0], 10))
    y_train_one_hot[[i for i in range(y_train.shape[0])], y_train] = 1
    print('Preprocession completed.')
    return x_train, y_train, y_train_one_hot, x_test, y_test
