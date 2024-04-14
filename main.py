import data_preprocess
from network import BP_network
import data_preprocess as d_p
import matplotlib.pyplot as plt


def main():
    pass


def test():
    """
    # print(x_train[0], x_test[0])
    layers = [(784, 256, 'sigmoid'), (256, 100, 'sigmoid'), (100, 10, 'softmax')]
    """
    model = BP_network(layers=[(784, 256, 'sigmoid'), (256, 100, 'sigmoid'), (100, 10, 'softmax')], model_path='test02.pkl')
    x_train, y_train, y_train_one_hot, x_test, y_test = data_preprocess.train_treat()
    model.train_and_evaluate(x_train, y_train_one_hot, y_train, x_test[:2000], y_test[:2000])
    model.test(x_test, y_test)
    model.save_model()
    '''fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(10):
        ax[i].imshow(x[i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()'''



if __name__ == "__main__":
    test()
