import time

import img_preprocess_and_cache as i_p
from network import BP_network


def debug1():
    model = BP_network(load_from_trained=True, model_path='3_layers.pkl')
    pics = ['20240417212258', '20240417232801', '20240418000115']
    servive = i_p.img_administration('default')
    for pic in pics:
        _, digits = servive.fetch('test_imgs/' + pic + '.jpg')
        print(model.predict(digits))
        time.sleep(1.2)


# 第二组图片从左到右依次均为'1, 2, 3, 4, 5, 6, 7, 8, 9, 0'
def debug2():
    model = BP_network(load_from_trained=True, model_path='3_layers.pkl')
    servive = i_p.img_administration('default')
    ok = 0
    for i in range(10):
        print('\n', i + 1)
        _, digits = servive.fetch(f'test_imgs/{i + 1}.jpg')
        if len(digits) == 10:
            ok += 1
        print(model.predict(digits))
        time.sleep(1.2)
    print('OK:', ok)


if __name__ == '__main__':
    debug2()
