import os
import importlib, inspect
from utils import recursive_vars

class_dict = {}

for file in os.listdir('models'):
    if file.endswith('.py') and file != '__init__.py':
        module = importlib.import_module(f'.{os.path.splitext(file)[0]}', package='models')
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__:
                class_dict[name] = cls



def load_model_from_cfg(cfg):
    model_setting = recursive_vars(cfg.model.model_setting)

    using_model_setting = {}
    for k, v in model_setting.items():
        if isinstance(v, str) and v.startswith('from:'):
            paths = v.strip('from:').split('.')
            cur_pos = cfg
            for cur in paths:
                cur_pos = getattr(cur_pos, cur)
            v = cur_pos
        using_model_setting[k] = v

    return load_model(cfg.model.model_class, using_model_setting)

def load_model(model_type, model_setting={}):
    if model_type not in class_dict:
        raise NotImplementedError()
    
    model_class = class_dict[model_type]
    return model_class(**model_setting)
#     elif model_cfg.model_class == 'Ithemal':
#         # model = 
#         rnn_params = RnnParameters(
#             embedding_size=512,
#             hidden_size=512,
#             num_classes=1,
#             connect_tokens=False,
#             skip_connections=False,
#             hierarchy_type=RnnHierarchyType.MULTISCALE,
#             rnn_type=RnnType.LSTM,
#             learn_init=False,
#             pad_idx=model_cfg.pad_idx,
#         )
#         md = RNN(rnn_params)
#         md.set_learnable_embedding(mode='none', dictsize=model_cfg.vocab_size)
#         return md
#     elif model_cfg.model_class == 'BatchIthemal':
#         # model = 
#         rnn_params = RnnParameters(
#             embedding_size=model_cfg.dim,
#             hidden_size=model_cfg.dim,
#             num_classes=1,
#             connect_tokens=False,
#             skip_connections=False,
#             hierarchy_type=RnnHierarchyType.MULTISCALE,
#             rnn_type=RnnType.LSTM,
#             learn_init=False,
#             pad_idx=model_cfg.pad_idx,
#         )
#         md = BatchRNN(rnn_params)
#         md.set_learnable_embedding(mode='none', dictsize=model_cfg.vocab_size)
#         return md
#     elif model_cfg.model_class == 'BatchRNN2':
#         # model = 
#         rnn_params = RnnParameters(
#             embedding_size=model_cfg.dim,
#             hidden_size=model_cfg.dim,
#             num_classes=1,
#             connect_tokens=False,
#             skip_connections=False,
#             hierarchy_type=RnnHierarchyType.MULTISCALE,
#             rnn_type=RnnType.LSTM,
#             learn_init=False,
#             pad_idx=model_cfg.pad_idx,
#         )
#         md = BatchRNN2(rnn_params)
#         md.set_learnable_embedding(mode='none', dictsize=model_cfg.vocab_size)
#         return md


