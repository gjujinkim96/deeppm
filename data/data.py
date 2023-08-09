#main data file

import numpy as np
import data.utilities as ut
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


class Data(object):

    """
    Main data object which extracts data from a database, partition it and gives out batches.

    """
    def __init__(self): #copy constructor
        self.percentage = 80
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

    def generate_datasets(self, hyperparameter_test=False, hyperparameter_test_mult=0.2, 
                        short_only=False):
        size = len(self.data)
        split = (size * self.percentage) // 100

        self.test_idx = []
        if short_only:
            self.train = []
            train_cnt = 0
            self.test = []

            for idx, datum in enumerate(self.data):
                if datum.block.num_instrs() < 50 and train_cnt < split:
                    self.train.append(datum)
                    train_cnt += 1
                else:
                    self.test.append(datum)
                    self.test_idx.append(idx)
        else:
            self.train  = self.data[:split]
            self.test = self.data[split:]
            self.test_idx = list(range(split, len(self.data)))

        if hyperparameter_test:
            self.train = self.train[:int(len(self.train) * hyperparameter_test_mult)]
            self.test = self.test[:int(len(self.test) * hyperparameter_test_mult)]
        print ('train ' + str(len(self.train)) + ' test ' + str(len(self.test)))
