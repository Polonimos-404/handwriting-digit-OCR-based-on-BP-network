import pickle
from os import makedirs
from statistics import mean, stdev
from time import strftime, localtime

import cv2
import numpy as np
from matplotlib import pyplot as plt

HISTORY_FOLDER = 'history/'  # 历史记录文件夹
IMAGE_NPZ_FOLDER = 'img_npzs/'  # 一次图片识别进程中，转换为(28, 28)大小的ndarray数组后的图片缓存文件夹
MARKED_IMAGE_NAME = 'img_marked.jpg'  # 一次图片识别进程中，标记出每个数字轮廓的图片文件名
GLOBAL_IMAGE_SIZE = 28  # MNIST数据集图片标准边长


class img_administration:
    """
    图片管理类
    说明：
        识别中的一个子过程，是获取图片->对图片进行处理->返回经过处理的数据
        出于实际应用中调取历史记录的需要，需要将一定量已处理的数据缓存于本地磁盘中
        因此该类主要实现的是对图片的处理和历史记录缓存
        历史记录的文件结构：
            ./history/
                history_name1.pkl
                history_name2.pkl
                ...
                history_name1/
                    cur_time1/
                        img_marked
                        img_npzs/
                            1.npz
                            ...
                    cur_time2/
                    ...
                history_name2/
                ...
    """

    def __init__(self, history_name: str = 'default'):
        """

        :param history_name: 历史记录文件名&对应文件夹名。历史纪录默认采用pickle存储，为字典格式，一个元素内容为：
                           key=img_path, val=(cur_time, dgt_cnt)
                           img_path: 本次传入的图片文件路径
                           cur_time: 处理时间（在调用image_read_and_treat时获取），格式为'YYYYmmdd_HHMMSS'（如'20240419_195458'）
                           dgt_cnt: 图片中包含的数字个数
        """
        self.history_name = history_name
        self.path_match = {}  # 历史记录数据
        self.load_match()

    def fetch(self, img_path: str):
        """
        传回所需数据（如前所述）。如果img_path在缓存中，直接从相应路径中调取，否则由img_read_and_treat返回
        :param img_path: 要识别的图片文件路径（绝对路径）
        :return: digit_arrays: 处理后每个数字对应的ndarray数组
        """
        digit_arrays = []
        if img_path in self.path_match:
            # 导航至目标路径，按数字逐个读取
            (this_time, dgt_cnt) = self.path_match[img_path]
            dirc = HISTORY_FOLDER + self.history_name + '/' + this_time + '/'
            img_marked = cv2.imread(dirc + MARKED_IMAGE_NAME)
            for i in range(dgt_cnt):
                with np.load(dirc + IMAGE_NPZ_FOLDER + str(i + 1) + '.npz', allow_pickle=True) as f:
                    array = f['arr_0']
                digit_arrays.append(array)
        else:
            img_marked, digit_arrays = self.image_read_and_treat(img_path)
        # 展示处理后的数字灰度图和标记出数字轮廓的原图
        show(digit_arrays)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        cv2.imshow('img', img_marked)
        cv2.waitKey(0)
        return img_marked, digit_arrays

    # 加载历史记录
    def load_match(self):
        try:
            with open(HISTORY_FOLDER + self.history_name + '.pkl', 'rb') as f:
                self.path_match = pickle.load(f)
        except FileNotFoundError:
            pass  # print('Cache file not found. New file will be created.')

    # 更新和存储历史记录
    def save_cache(self, img_path: str, cur_time: str, dgt_cnt: int):
        self.path_match[img_path] = (cur_time, dgt_cnt)
        with open(HISTORY_FOLDER + self.history_name + '.pkl', 'wb') as f:
            pickle.dump(self.path_match, f)

    def image_read_and_treat(self, img_path: str):
        """
        图片读取和预处理。将实际图片：1.二值化
                               2.分割为单个数字图像
                               3.分别缓存为标记出各个数字轮廓的图片和对应每个数字的(784, )ndarray数组
        :param img_path: 要读取的文件路径（一般为绝对路径）
        :return: img_marked: ndarray格式，从左至右第奇数个图片轮廓标记为蓝色，第偶数个图片轮廓标记为红色
                 digits_array: list格式，每个元素为一个数字的ndarray数组
        """
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        # 转换为灰度图，并二值化（findContours只能处理二值图）
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 使用dilate()对图片进行膨胀处理，以消除可能出现的数字断裂问题
        img_bin = np.rot90(img_bin)  # 之后findContours()检测轮廓的顺序为从下到上，故需要先将图片逆时针旋转90°
        # show(img_bin)
        kernel_connect = np.ones((3, 3), np.uint8)  # 膨胀操作使用的内核
        ib_dilated = cv2.dilate(img_bin, kernel_connect, iterations=2)

        # 使用findContours()检测图片中每个数字的轮廓
        ib_dilated_2 = ib_dilated.copy().astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(ib_dilated_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # contours：list结构，列表中每个元素代表一个边沿信息。每个元素是(x, 1, 2)的三维向量，
        # x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标；
        # hierarchy：返回类型是(x, 4)的二维ndarray。x和contours里的x是一样的意思。
        # 如果输入选择cv2.RETR_TREE，则以树形结构组织输出，
        # hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。

        # 用标准差法剔除大小明显离群的轮廓，即非数字的轮廓。sigma默认值为3
        def thr_sigma(dataset, n=2.5):
            d_mean = mean(dataset)  # 得到均值
            sigma = stdev(dataset, d_mean)  # 得到标准差
            remove_idx = [idx for idx, size in enumerate(dataset) if abs(size - d_mean) > n * sigma]
            return remove_idx

        sizes = [sub.shape[0] for sub in contours]
        illegal = thr_sigma(sizes)
        if illegal:
            for i in illegal:
                del contours[i]

        '''for it in contours:
            print([it])
        print("##########################")'''

        cur_time = strftime('%Y%m%d_%H%M%S', localtime())  # 获取时间
        # 创建下层缓存文件夹
        makedirs(HISTORY_FOLDER + self.history_name + '/' + cur_time + '/' + IMAGE_NPZ_FOLDER.strip('/'))
        img_marked = np.rot90(img)
        digit_arrays = []
        for i, contour in enumerate(contours):
            [x, y, w, h] = cv2.boundingRect(contour)  # 当得到数字轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形，
            # show(cv2.boundingRect(contour))

            img_marked[y:y + h, x:x + w, (i % 2) * 2] = 255

            # 取出图片中对应当前数字的部分
            digit = ib_dilated[y:y + h, x:x + w]
            # show(digit)

            # 利用copyMakeBorder()增加边框，将digit扩充为正方形
            pad_len = (h - w) // 2
            # print(pad_len)
            if pad_len > 0:
                digit = cv2.copyMakeBorder(digit, 0, 0, pad_len, pad_len, cv2.BORDER_CONSTANT, value=0)
            elif pad_len < 0:
                digit = cv2.copyMakeBorder(digit, -pad_len, -pad_len, 0, 0, cv2.BORDER_CONSTANT, value=0)

            pad = digit.shape[0] // 4  # 避免数字与边框直接相连，留出4个像素左右
            digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            digit = cv2.resize(digit, (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)  # 把图片缩放至28*28
            digit = np.rot90(digit, 3)  # 逆时针旋转270°将图片旋转为原来的水平方向
            # show(digit)
            # 接下来与数据集处理的方法相同
            digit = (digit / 255 - 0.1307) / 0.3081
            digit = digit.reshape(GLOBAL_IMAGE_SIZE ** 2)
            np.savez(HISTORY_FOLDER + self.history_name + '/' + cur_time + '/' + IMAGE_NPZ_FOLDER + str(i + 1) + '.npz',
                     digit)
            digit_arrays.append(digit)

        img_marked = np.rot90(img_marked, 3)
        cv2.imwrite(HISTORY_FOLDER + self.history_name + '/' + cur_time + '/' + MARKED_IMAGE_NAME, img_marked)
        print('INFO: 图像预处理已完成.')
        self.save_cache(img_path, cur_time, len(digit_arrays))
        return img_marked, digit_arrays


def show(sub=None):
    """

    :type sub: list[np.ndarray]
    """
    imgs = sub.copy()
    size = len(imgs)
    row = size // 5 + int(size % 5 != 0)
    plt.figure(dpi=300)
    for i in range(size):
        plt.subplot(row, 5, i + 1)
        plt.imshow(imgs[i].reshape(GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE), cmap='Greys', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
