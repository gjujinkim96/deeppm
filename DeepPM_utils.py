import sys
import os

from enum import Enum
import torch
from typing import Any, NamedTuple

import data.data_cost as dt
#import models.graph_models as md

def dump_idx_to_root(expt, data):
    idx_dict = {
        'train': [datum.code_id for datum in data.train],
        'val': [datum.code_id for datum in data.val],
        'test': [datum.code_id for datum in data.test],
    }

    dump_obj_to_root(expt, idx_dict, 'idx_dict.dump')

def dump_obj_to_root(expt, obj, name):
    try:
        os.makedirs(expt.experiment_root_path()) 
    except OSError:
        pass

    torch.save(obj, os.path.join(expt.experiment_root_path(), name))

def load_model_and_data(fname):
    # type: (str) -> (md.AbstractGraphMode, dt.DataCost)
    dump = torch.load(fname)
    data = dt.DataInstructionEmbedding()
    data.read_meta_data()
    data.load_dataset_params(dump.dataset_params)
    return (dump.model, data)

