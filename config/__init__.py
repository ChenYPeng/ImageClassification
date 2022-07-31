import os
import yaml


def get_config(config_name):
    config_path = os.path.join('config', config_name)
    file = open(config_path, 'r', encoding="utf-8")
    datas = yaml.load(file, Loader=yaml.FullLoader)
    return datas