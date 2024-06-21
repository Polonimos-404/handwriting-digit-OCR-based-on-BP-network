from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from network import List, BP_network, MODELS_FOLDER


# K折交叉验证，用于选择最优超参数
def k_fold_cross_validation(x, y, y_one_hot,
                            hyperparameters: List[List[tuple]] = None, paths: List[str] = None,
                            k=10, epochs=50, learning_rate=1e-3, batch_size=60):
    """

    :param hyperparameters: 传入的自定义超参数列表，其中每个List代表一个模型，每个List中一个元组代表一层神经元
    :param paths: 由models文件夹导入的超参数列表
    :param x: 未分割的训练集
    :param y:
    :param y_one_hot:
    :param k: 折数
    :param epochs:
    :param learning_rate:
    :param batch_size:
    """

    models = []
    # 从自定义的超参数列表中创建模型
    if hyperparameters is not None:
        for i, net in enumerate(hyperparameters):
            models.append(BP_network(net, model_path=MODELS_FOLDER + f'authorized {i}.pkl'))
    # 获取已有的模型
    if paths is not None:
        for path in paths:
            models.append(BP_network(load_from_trained=True, model_path=MODELS_FOLDER + path + '.pkl'))

    avg_scores = []
    for model in models:
        # 利用sklearn中的StratifiedKFold将数据集分为k折
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        scores = []
        for train_idx, valid_idx in skf.split(x, y):
            x_train, y_train_one_hot, y_train = x[train_idx], y_one_hot[train_idx], y[train_idx]
            x_validate, y_validate = x[valid_idx], y[valid_idx]
            model.train_and_evaluate(x_train, y_train_one_hot, epochs=epochs, learning_rate=learning_rate,
                                     batch_size=batch_size, trace=False, draw=False)
            acc_score = accuracy_score(y_validate, model.predict(x_validate))
            scores.append(acc_score)
        avg_score = sum(scores) / k
        avg_scores.append(avg_score)

    # 选择平均准确率最高的模型
    mx_sc = 0.0
    mx_sc_idx = -1
    print('Average accuracy scores of each model:')
    for i, score in enumerate(avg_scores):
        print(f'{models[i].model_path}    {score}')
        if score > mx_sc:
            mx_sc = score
            mx_sc_idx = i
    print(f'\nModel with highest accuracy score: {models[mx_sc_idx].model_path}, {mx_sc * 100}%')
    # 如果最佳模型来源于自定义超参数列表，则将其保存
    if hyperparameters is not None and mx_sc_idx < len(hyperparameters):
        models[mx_sc_idx].save_model()
