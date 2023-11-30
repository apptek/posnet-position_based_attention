import os
import importlib


__MODEL_DICT__ = dict()

def create_model_from_config(config, vocab_src, vocab_tgt):

    return __MODEL_DICT__[config['model']].create_model_from_config(config, vocab_src, vocab_tgt)


def register_model(name):
    
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('models.' + module_name)