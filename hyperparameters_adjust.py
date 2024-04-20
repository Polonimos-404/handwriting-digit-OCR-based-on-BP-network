from models.network import List, BP_network
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


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
    folder = 'models'
    models = []
    if hyperparameters is not None:
        for i, net in enumerate(hyperparameters):
            models.append(BP_network(net, model_path=folder + f'/authorized {i}.pkl'))
    if paths is not None:
        for path in paths:
            models.append(BP_network(load_from_trained=True, model_path=folder + path + '.pkl'))

    avg_scores = []
    for i, model in enumerate(models):
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

    mx_sc = 0.0
    mx_sc_idx = -1
    print('Average accuracy scores of each model:')
    for i, score in enumerate(avg_scores):
        print(f'{models[i].model_path}    {score}')
        if score > mx_sc:
            mx_sc = score
            mx_sc_idx = i
    print(f'\nModel with highest accuracy score: {models[mx_sc_idx].model_path}, {mx_sc * 100}%')
    if hyperparameters is not None and mx_sc_idx < len(hyperparameters):
        models[mx_sc_idx].save_model()
