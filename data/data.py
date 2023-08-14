#main data file

import numpy as np
import data.utilities as ut
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


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

    def generate_datasets(self, split_mode='none', hyperparameter_test=False, hyperparameter_test_mult=0.2, split_perc=(8, 2, 0)):
        def get_train_val(size, train, val):
            total_perc = train + val
            train_size = int(size * (train / total_perc))
            val_size = size - train_size
            return train_size, val_size
        
        tmp = self.data

        self.test = []

        if split_perc[2] > 0:
            random.shuffle(tmp)
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

            group_limit = [5, 10, 23, 50, 100, 150, 200]
            def get_group(x):
                for idx, limit in enumerate(group_limit):
                    if x < limit:
                        return idx
                return idx + 1


            for datum in tmp:
                g_type = get_group(datum.block.num_instrs())
                g[g_type].append(datum)

            for k, grouped_data in g.items():
                random.shuffle(grouped_data)

                train_size, val_size = get_train_val(len(grouped_data), split_perc[0], split_perc[1])
                self.train.extend(grouped_data[:train_size])
                self.val.extend(grouped_data[train_size:])
        elif split_mode == 'num_instrs+srcs':
            self.train = []
            self.val = []

            g = defaultdict(list)

            group_limit = [5, 10, 23, 50, 100, 150, 200]
            def get_group_by_num_instrs(x):
                for idx, limit in enumerate(group_limit):
                    if x < limit:
                        return idx
                return idx + 1
            
            def get_group(datum):
                num_instrs_group = get_group_by_num_instrs(datum.block.num_instrs())
                return datum.src, num_instrs_group


            for datum in tmp:
                g_type = get_group(datum)
                g[g_type].append(datum)

            for k, grouped_data in g.items():
                random.shuffle(grouped_data)

                train_size, val_size = get_train_val(len(grouped_data), split_perc[0], split_perc[1])
                self.train.extend(grouped_data[:train_size])
                self.val.extend(grouped_data[train_size:])
        else:
            raise NotImplementedError()

        if hyperparameter_test:
            self.train = self.train[:int(len(self.train) * hyperparameter_test_mult)]
            self.val = self.val[:int(len(self.val) * hyperparameter_test_mult)]
            self.test = self.test[:int(len(self.test) * hyperparameter_test_mult)]

        saying = f'train: {len(self.train)}  val: {len(self.val)}'
        if len(self.test) > 0:
            saying += f'  test: {len(self.test)}'
        
        print(saying)
