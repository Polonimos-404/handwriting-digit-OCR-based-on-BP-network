import numpy as np
import pickle as pkl
from typing import List
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss, accuracy_score, precision_score, f1_score, recall_score
from time import perf_counter


np.random.seed(0)


# 单层神经元
class network_layer:
    def __init__(self, input_size: int = None, output_size: int = None, _weights=None, _bias=None,
                 _activation: str = None):
        """

        :param int input_size: 输入神经元数
        :param int output_size: 输出神经元数
        :param str _activation: 激活函数类型
        :param _weights: 权重矩阵
        :param _bias: 偏置向量
        """
        self.weights = _weights if _weights is not None \
            else np.random.uniform(-2.4 / input_size, 2.4 / input_size, (input_size, output_size))
        self.bias = _bias if _bias is not None else np.random.rand(output_size) * 0.25
        self.activation = _activation  # 激活函数类型
        self.activation_output = None  # 激活函数输出
        self.error = None  # 用于计算delta的中间变量
        self.delta = None  # delta变量，即梯度

    # 向前传播函数
    def activate(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        self.activation_output = self._apply_activation(z)  # 获得并记录激活函数输出
        return self.activation_output

    # 激活函数处理
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation is None:
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':  # f(zi) = exp(zi) / sum(exp(z))
            output = np.exp(x)
            return output / np.sum(output)
        return x

    # 激活函数对应导数处理
    def apply_derivative(self, x: np.ndarray) -> np.ndarray:
        # 传入参数x为当前层激活函数f(z)输出
        if self.activation is None:
            return np.ones_like(x)
        elif self.activation == 'sigmoid':  # f'(z) = f(z)[1 - f(z)]
            return x * (1 - x)
        elif self.activation == 'relu':  # f'(z) = 0 when z < 0; 1 when z >= 0
            grad = np.zeros_like(x)
            grad[x > 0] = 1
            return grad
        elif self.activation == 'tanh':  # f'(z) = 1 - f(z) ^ 2
            return 1 - x ** 2
        return x


# BP神经网络
class BP_network:
    def __init__(self, layers: List[tuple] = None, load_from_trained=False, model_path: str = None):
        """

        :param layers: 每个元组代表一层，其结构为(input_size, output_size, activation)
        :param load_from_trained: 是否从models文件夹中加载
        :param model_path: 模型文件对应路径
        """
        self.layers = []
        self.model_path = model_path  # 每个模型独有的存储路径
        if load_from_trained:
            self.load_model()
        elif layers is not None:
            self.layers = [network_layer(in_sz, out_sz, _activation=act) for (in_sz, out_sz, act) in layers]  # 层对象列表

        self.layers_count = len(self.layers)
        print(f'Model created, {self.layers_count} layers in total.')

    # 从指定路径中加载预训练模型
    def load_model(self):
        print('Loading model...')
        try:
            with open(self.model_path, 'rb') as f:
                layer_parameters = pkl.load(f)
                self.layers = [network_layer(_weights=w, _bias=b, _activation=act) for (w, b, act) in layer_parameters]

            print('Model loaded from "' + self.model_path + '".')
        except FileNotFoundError:
            print('Path does not exist.')

    # 保存当前模型
    def save_model(self):
        print('Saving model...')
        with open(self.model_path, 'wb') as f:
            # 使用三元组(weights, bias, activation)表示一层的参数
            layer_parameters = [(layer.weights, layer.bias, layer.activation) for layer in self.layers]
            pkl.dump(layer_parameters, f)
        print('Model saved as "' + self.model_path + '".')

    # 添加层
    def append_layers(self, i: int, new_layers: List[network_layer]):
        for layer in reversed(new_layers):
            self.layers.insert(i, layer)
        self.layers_count += len(new_layers)
        print('New layers added.')
        self.show_info()

    # 逐层打印网络信息
    def show_info(self):
        if self.layers_count == 0:
            print('Empty network. Please append layers first.')
        else:
            # 一个层的基本参数只有三个：Input Channels（输入维数），Output Channels（输出维数），Activation（激活函数类型）
            print(f'Total layers count: {self.layers_count}')
            layer_info = [[layer.weights.shape[0], layer.bias.shape[0], layer.activation] for layer in self.layers]
            mtx = DataFrame(layer_info, index=[f'Layer{i}' for i in range(1, self.layers_count + 1)],
                            columns=['Input Channels', 'Output Channels', 'Activation'])
            print(mtx)

    # 向前传播
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.activate(x)
        return x

    # 反向传播的实现
    def back_propagation(self, x_batch, y_batch, learning_rate, batch_size):
        """

        :param x_batch:
        :param y_batch:
        :param learning_rate: 学习率
        :param batch_size: 一个batch中含有的样本个数
        :return:
        """
        # 定义一次更新中w、b的累计偏导矩阵（每次使用一个batch的平均梯度来更新参数）
        nabla_w = [np.zeros_like(layer.weights) for layer in self.layers]
        nabla_b = [np.zeros_like(layer.bias) for layer in self.layers]
        for x, y in zip(x_batch, y_batch):
            output = self.feed_forward(x)
            # 反向逐层计算梯度
            # 输出层
            output_layer = self.layers[-1]
            output_layer.delta = y - output  # 损失函数为多分类交叉熵损失函数
            nabla_w[-1] += output_layer.delta * np.atleast_2d(self.layers[-2].activation_output).T
            nabla_b[-1] += output_layer.delta
            next_layer = output_layer
            # 其他层
            for i in range(2, self.layers_count + 1):
                layer = self.layers[-i]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_derivative(layer.activation_output)
                # 累加梯度
                nabla_w[-i] += (layer.delta *
                                np.atleast_2d(x if i == self.layers_count else self.layers[-i - 1].activation_output).T)
                nabla_b[-i] += layer.delta
                next_layer = layer

        # 正向逐层更新参数
        for i, layer in enumerate(self.layers):
            layer.weights += nabla_w[i] * learning_rate / batch_size
            layer.bias += nabla_b[i] * learning_rate / batch_size

    # 训练（和评估）函数
    def train_and_evaluate(self, x_train, y_train_one_hot, y_train=None, x_test0=None, y_test0=None,
                           epochs=30, learning_rate=0.1, batch_size=60, trace=False, draw=False):
        """

        :param x_train: 训练样本特征
        :param y_train_one_hot: 训练样本标签(one_hot化)
        :param y_train: 训练样本标签(取值0-9)
        :param x_test0: 验证样本集，可以是训练集或测试集中分出的一部分
        :param y_test0:
        :param epochs: 训练次数
        :param learning_rate: 学习率
        :param batch_size: 一个batch的大小。== 1时相当于随机梯度下降(SGD)，> 1时相当于小批量梯度下降(MBGD)，== len(x_train)时相当于批量梯度下降(BGD)
        :param trace: 是否跟踪训练情况。若为True，将输出和暂存每轮训练后模型预测结果在训练集上的交叉熵损失、准确率和在验证集上的准确率
        :param draw: 是否绘制上述三个评价参数随训练轮数增加而变化的图线
        **在trace == True时实际上不需要传入x_test0, y_test0**
        """
        if self.layers_count < 2:
            print('Layers insufficient, the network should contain AT LEAST 3 LAYERS.\nPlease append layers first.')
        else:
            print('Training begins.')
            # 训练结束时输出训练所经历的时间
            start_t = perf_counter()  # 计时起始时间（时间戳）
            # 打印当前网络信息
            # print('Current network info:')
            # self.show_info()

            losses, acc_trn, acc_tst = [], [], []
            for i in range(epochs):
                perm = np.random.permutation(y_train.shape[0])
                x_train = x_train[perm, :]
                y_train_one_hot, y_train = y_train_one_hot[perm], y_train[perm]
                # 分批次(batch)训练
                for j in range(0, len(x_train), batch_size):
                    x_batch, y_batch = x_train[j: j + batch_size], y_train_one_hot[j: j + batch_size]
                    self.back_propagation(x_batch, y_batch, learning_rate, batch_size)

                # 训练过程中每轮记录损失和准确率
                if trace:
                    cross_entropy_loss = log_loss(y_train_one_hot, self.feed_forward(x_train))
                    accuracy_train = accuracy_score(y_train, self.predict(x_train))
                    accuracy_test = accuracy_score(y_test0, self.predict(x_test0))
                    print(
                        "\nEpoch: {:d}\nCEL on train set: {:.4f}\nAccuracy on train set: {:.2f}%\nAccuracy on test set: {:.2f}%"
                        .format(i + 1, cross_entropy_loss, accuracy_train * 100, accuracy_test * 100))
                    losses.append(cross_entropy_loss)
                    acc_trn.append(accuracy_train)
                    acc_tst.append(accuracy_test)

            end_t = perf_counter()  # 计时终止时间
            scd_elp = end_t - start_t
            print('Training Completed.')
            print(f'Time elapsed: {int(scd_elp // 60)}min{scd_elp % 60}s')

            # 绘制loss和accuracy关于epoch的变化图线
            if draw:
                epoch = [i for i in range(epochs)]
                plt.subplot(2, 1, 1)
                plt.plot(epoch, losses, label='loss', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Cross Entropy Loss    lr={learning_rate}')
                plt.subplot(2, 1, 2)
                plt.plot(epoch, acc_trn, label='train_acc', marker='*')
                plt.plot(epoch, acc_tst, label='test_acc', marker='>')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'Train Acc vs Test Acc    lr={learning_rate}')
                plt.legend()
                plt.show()

        # 训练完成直接保存模型
        self.save_model()

    # 给出一定特征集上的预测
    def predict(self, x_predict: np.ndarray or List):
        return np.array([np.argmax(self.feed_forward(x)) for x in x_predict])

    # 测试函数
    def test(self, x_test, y_test):
        print('Test processing...')
        results = self.predict(x_test)
        # 获得并打印混淆矩阵
        cm = confusion_matrix(y_test, results)
        conf_mtx = DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
        print('______Results______')
        print('------Confusion Matrix------')
        print(conf_mtx)
        print('Accuracy: ', accuracy_score(y_test, results))    # 在整个测试集上的准确率
        # 使用宏平均(Macro Avg)，加权平均(Weighted Avg)和微平均(Micro Avg)三种方式计算准确率，召回率和F1分数三种评估指标
        # 加权平均计算方法与微平均类似，只是在求各个基本参量平均值时以各类样本的占比作为权重进行加权
        evaluate_metrics = []
        for metric_type in ['weighted', 'macro', 'micro']:
            p_s = precision_score(y_test, results, average=metric_type)
            r_s = recall_score(y_test, results, average=metric_type)
            f_s = f1_score(y_test, results, average=metric_type)
            evaluate_metrics.append([p_s, r_s, f_s])
        evaluate_results = DataFrame(evaluate_metrics, index=['Weighted', 'Macro', 'Micro'],
                                     columns=['Precision', 'Recall', 'F1_Score'])
        print('------Evaluation Metrics------')
        print(evaluate_results)
