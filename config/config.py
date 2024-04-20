import pickle
from os.path import exists

"""
    这是一个可供参考的配置（全局变量）管理模块
"""


DEFAULT_CONFIG = '_default'  # 默认配置文件
CACHE_CONFIG = 'cache'  # 缓存配置文件，用于保存上一次修改后的配置
CUSTOM_CONFIG_FOLDER = 'custom/'    # 用户自定义的配置文件夹（如果用户有将某一套配置专门存储的需要）
GLOBAL_VARIBLES = {}    # 存储全部全局变量的字典


"""
文件结构
    config/
        config.py
        _default.pkl
        cache.pkl
        custom/
            custom1.pkl
            ...
"""


# 从缓存的配置中初始化全局变量，在程序启动时调用
def _init():
    load_config_from(CACHE_CONFIG)


# 获取某一全局变量的值
def get_val(name: str):
    try:
        return GLOBAL_VARIBLES[name]
    except KeyError:
        return "Not Found"


# 修改某一全局变量的值（如果发生了修改必须在主函数中调用save_cache，以实现对修改的记忆）
def set_val(name: str, val):
    if name in GLOBAL_VARIBLES:
        GLOBAL_VARIBLES[name] = val
        return True
    else:
        return False


# 存储专门的配置文件
def save_custom(filename: str):
    p = CUSTOM_CONFIG_FOLDER + filename
    if exists(p):
        print("Filename already exists, do you want to overwrite?")
        sel = input()
        if sel == 'N':
            p += '(1)'

    save_config_to(p)


# 从配置文件中加载配置（如果调用了该函数，必须在主函数中调用save_cache）
def load_custom(path: str):
    if exists(path):
        load_config_from(path)


# 对修改后的配置进行缓存。（如果运行过程中发生了对配置的修改）在退出程序时调用
def save_cache():
    save_config_to(CACHE_CONFIG)


# 恢复至默认配置
def restore_to_default():
    load_config_from(DEFAULT_CONFIG)
    save_config_to(CACHE_CONFIG)


def save_config_to(path: str):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(GLOBAL_VARIBLES, f)


def load_config_from(path: str):
    global GLOBAL_VARIBLES
    with open(path + '.pkl', 'rb') as f:
        GLOBAL_VARIBLES = pickle.load(f)
