import os 
import yaml
import logging
from easydict import EasyDict


# read yaml config file
def get_config(config_path):

    # read yaml file, load parameters
    with open(config_path, 'r') as f:

        config = yaml.safe_load(f)
        config = EasyDict(config)
    
    return (config)


# set logger handle
def set_logger(config):

    # create logger handle
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s-%(filename)s-%(levelname)s]-%(message)s')

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file handler
    file_handler = logging.FileHandler(os.path.join(config['path'], config['name']), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return (logger)

