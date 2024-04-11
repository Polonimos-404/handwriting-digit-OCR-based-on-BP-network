from network import BP_network
import data_preprocess


def main():
    pass


def test():
    x_train, y_train, y_train_one_hot, x_test, y_test = data_preprocess.treat()
    layers = [(784, 256, 'sigmoid'), (256, 100, 'sigmoid'), (100, 10, 'softmax')]
    tst_md = BP_network(layers, model_path='test01.pkl')
    tst_md.show_info()
    tst_md.train_and_evaluate(x_train, y_train_one_hot, y_train, x_test[:1000], y_test[:1000], trace=True, draw=True)
    tst_md.test(x_test, y_test)
    tst_md.save_model()


if __name__ == "__main__":
    test()
