import os
import numpy as np
from typing import List
from skimage import io, transform
import time

"""
    必须要做的:
    1. 把单个数字的图片转换为28 * 28的灰度图像，缓存
    2. 图形化操作界面：
        至少包含File，About两个部分
    可能能做的：
    1. 把含有多个数字的图像分开，逐个识别
    2. 允许用户在自定义的数据集上对模型进行微调/训练新的模型（允许自定义各个参数，提供进度显示）
        从而允许用户使用自由选择模型进行预测
"""

global_discrete_image_folder = 'source/cache/disct_img/'    # 分割后的图片缓存文件夹
global_translated_image_folder = 'source/cache/trans_img/'     # 处理为与训练集相同的ndarray，且存储为npz格式的图片缓存文件夹
global_dataset_folder = 'source/dataset/'   # 数据集文件夹

# 图片预处理：分割为单个数字图像
'''
def image_divide(path: str):
'''


# 图片预处理：格式转换
def image_translate_and_save(paths: List[str]):
    # cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    imgs = []
    for path in paths:
        img = io.imread(global_discrete_image_folder + path, as_gray=True)
        img = transform.resize(img, (28, 28), anti_aliasing=True).reshape(28 * 28)
        img = (1 - img - 0.1307) / 0.3081
        imgs.append(img)
        p = global_translated_image_folder + path + '.npz'
        np.savez(p, img)
    return imgs


def train_treat(path: str = global_dataset_folder + 'mnist.npz'):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    # 将特征集每个数据扁平化为28 * 28 = 784维行向量，并归一化（最大最小归一+Z-score标准化）
    # 最大最小归一化后，MNIST数据集均值为0.1307，标准差为0.3081
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).astype(np.float32)
    x_train, x_test = (x_train / 255.0 - 0.1307) / 0.3081, (x_test / 255.0 - 0.1307) / 0.3081
    # 为了方便计算误差，另单独生成一份one_hot编码的训练集标签
    y_train_one_hot = np.zeros((y_train.shape[0], 10))
    y_train_one_hot[[i for i in range(y_train.shape[0])], y_train] = 1
    print('Preprocession completed.')
    return x_train, y_train, y_train_one_hot, x_test, y_test
