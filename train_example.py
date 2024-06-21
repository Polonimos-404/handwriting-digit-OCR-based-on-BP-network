from dataset_load import train_treat
from network import BP_network

# 这里提供一个训练模型的代码示例，实际开发过程中选择了多种不同超参数训练，
# 借助hyperparameters_adjust模块从多个模型中遴选得到当前选用的'3_layers.pkl'
x_train, y_train, y_train_one_hot, x_test, y_test = train_treat()
layers = [(784, 256, 'sigmoid'), (256, 100, 'sigmoid'), (100, 10, 'softmax')]
model = BP_network(layers, model_path='3_layers.pkl')
model.train_and_evaluate(x_train, y_train_one_hot, y_train, x_test[:2000], y_test[:2000],
                         epochs=50, learning_rate=0.005, batch_size=60, trace=True, draw=True)
model.test(x_test, y_test)
