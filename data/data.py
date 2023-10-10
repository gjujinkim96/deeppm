#main data file

import numpy as np
import data.utilities as ut
import random
from collections import defaultdict

def get_group(x):
    group_limit = [5, 10, 23, 50, 100, 150, 200]
    for idx, limit in enumerate(group_limit):
        if x < limit:
            return idx
    return idx + 1

class Data(object):

    """
    Main data object which extracts data from a database, partition it and gives out batches.

    """
    def __init__(self): #copy constructor
        self.costs = dict()

    def read_meta_data(self):

        self.sym_dict,_ = ut.get_sym_dict()
        self.offsets = ut.read_offsets()

        self.opcode_start = self.offsets[0]
        self.operand_start = self.offsets[1]
        self.int_immed = self.offsets[2]
        self.float_immed = self.offsets[3]
        self.mem_start = self.offsets[4]

        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = 1


    def generate_costdict(self, maxnum):
        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = np.random.randint(1,maxnum)

    def generate_datasets(self, split_mode='none',
                        split_perc=(8, 2, 0), shuffle=False, given_train_val_test_idx=None, small_size=False):
        
        if given_train_val_test_idx is not None:
            datum_mapping = {datum.code_id: datum for datum in self.data}
            self.train = [datum_mapping[idx] for idx in given_train_val_test_idx['train'] if idx in datum_mapping]
            self.val = [datum_mapping[idx] for idx in given_train_val_test_idx['val'] if idx in datum_mapping]
            self.test = [datum_mapping[idx] for idx in given_train_val_test_idx['test'] if idx in datum_mapping]

            if small_size:
                if len(self.val) == 0:
                    self.val = self.train[-2:]
                    self.train = self.train[:-2]
                
                if len(self.test) == 0:
                    self.test = self.train[-2:]
                    self.train = self.train[:-2]
            return 
        
        def get_train_val(size, train, val):
            total_perc = train + val
            train_size = int(size * (train / total_perc))
            val_size = size - train_size
            return train_size, val_size
        
        tmp = list(self.data)
        if shuffle:
            random.shuffle(tmp)

        self.test = []

        if split_perc[2] > 0:
            _, test_size = get_train_val(len(self.data), split_perc[0]+split_perc[1], split_perc[2])
            self.test.extend(tmp[:test_size])
            tmp = tmp[test_size:]

        self.train = []
        self.val = []
        if split_mode == 'none':
            train_size, val_size = get_train_val(len(tmp), split_perc[0], split_perc[1])
            self.train.extend(tmp[:train_size])
            self.val.extend(tmp[train_size:])
        elif split_mode == 'num_instrs':
            self.train = []
            self.val = []

            g = defaultdict(list)

            for datum in tmp:
                g_type = get_group(datum.block.num_instrs())
                g[g_type].append(datum)

            for k, grouped_data in g.items():
                random.shuffle(grouped_data)

                train_size, val_size = get_train_val(len(grouped_data), split_perc[0], split_perc[1])
                self.train.extend(grouped_data[:train_size])
                self.val.extend(grouped_data[train_size:])
        else:
            raise NotImplementedError()

        saying = f'train: {len(self.train)}  val: {len(self.val)}'
        if len(self.test) > 0:
            saying += f'  test: {len(self.test)}'
        
        print(saying)

    def mix_train_val(self, train_code_id, val_code_id, code_id_mapping):
        new_train = []
        for code_id in train_code_id:
            data_idx = code_id_mapping[code_id]
            new_train.append(self.data[data_idx])
        
        new_val = []
        for code_id in val_code_id:
            data_idx = code_id_mapping[code_id]
            new_val.append(self.data[data_idx])

        self.train = new_train
        self.val = new_val
