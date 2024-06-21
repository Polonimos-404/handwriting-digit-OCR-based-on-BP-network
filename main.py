from network import BP_network
from img_preprocess_and_cache import img_administration


def main():
    path = input('将欲识别的图片绝对路径粘贴于此处:')
    path = path.strip('"').replace('\\', '/')
    service = img_administration()
    img_marked, digit_arrays = service.fetch(path)
    model = BP_network(load_from_trained=True, model_path='3_layers.pkl')
    print('识别结果(从左至右): \n', model.predict(digit_arrays))


if __name__ == '__main__':
    main()
