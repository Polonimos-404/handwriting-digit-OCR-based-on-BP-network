from models.network import BP_network
from pic_cache import img_preprocess_and_cache as d_p
import time


def main():
    pass


def test():
    """
    print(x_train[0], x_test[0])
    layers = [(784, 256, 'sigmoid'), (256, 100, 'sigmoid'), (100, 10, 'softmax')]
    """
    model = BP_network(load_from_trained=True, model_path='default_mdl.pkl')
    # x_train, y_train, y_train_one_hot, x_test, y_test = d_p.train_treat()
    # model.train_and_evaluate(x_train, y_train_one_hot, y_train, epochs=20, learning_rate=0.01, batch_size=30)
    # model.test(x_test, y_test)
    pics = ['20240417212258', '20240417232801', '20240418000115']
    servive = d_p.img_administration('default.pkl')
    for pic in pics:
        _, digits = servive.fetch('test_imgs/' + pic + '.jpg')
        print(model.predict(digits))
        time.sleep(2)
    '''print(x_train[0], x_test[0])'''


if __name__ == "__main__":
    test()
