from .Config import Config

from .DeepPM import DeepPM
from .StackedDeepPM import StackedDeepPM
from .Stacked2dPosDeepPM import Stacked2dPosDeepPM
from .InstBlockOp import InstBlockOp
from .Pos2dInst import Pos2dInst
from .WithAttentionPooling import WithAttentionPooling
from .StackedDeepPM044 import StackedDeepPM044
from .StackedDeepPM008 import StackedDeepPM008
from .StackedDeepPM800 import StackedDeepPM800
from .StackedDeepPM080 import StackedDeepPM080
from .StackedDeepPM440 import StackedDeepPM440
from .StackedDeepPMPadZero import StackedDeepPMPadZero
from .Ithemal import RNN, RnnParameters, RnnHierarchyType, RnnType

def load_model(model_cfg):
    model = None
    if model_cfg.model_class == 'DeepPM':
        model = DeepPM
    elif model_cfg.model_class == 'StackedDeepPM':
        model = StackedDeepPM
    elif model_cfg.model_class == 'Stacked2dPosDeepPM':
        model = Stacked2dPosDeepPM
    elif model_cfg.model_class == 'InstBlockOp':
        model = InstBlockOp
    elif model_cfg.model_class == 'Pos2dInst':
        model = Pos2dInst
    elif model_cfg.model_class == 'WithAttentionPooling':
        model = WithAttentionPooling
    elif model_cfg.model_class == 'StackedDeepPM044':
        model = StackedDeepPM044
    elif model_cfg.model_class == 'StackedDeepPM008':
        model = StackedDeepPM008
    elif model_cfg.model_class == 'StackedDeepPM800':
        model = StackedDeepPM800
    elif model_cfg.model_class == 'StackedDeepPM080':
        model = StackedDeepPM080
    elif model_cfg.model_class == 'StackedDeepPM440':
        model = StackedDeepPM440
    elif model_cfg.model_class == 'StackedDeepPMPadZero':
        model = StackedDeepPMPadZero
    elif model_cfg.model_class == 'Ithemal':
        # model = 
        rnn_params = RnnParameters(
            embedding_size=512,
            hidden_size=512,
            num_classes=1,
            connect_tokens=False,
            skip_connections=False,
            hierarchy_type=RnnHierarchyType.MULTISCALE,
            rnn_type=RnnType.LSTM,
            learn_init=False,
        )
        md = RNN(rnn_params)
        md.set_learnable_embedding(mode='none', dictsize=628)
        return md
    else:
        raise NotImplementedError()
    
    return model(model_cfg)