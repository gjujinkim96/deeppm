#main data file

import numpy as np
import utilities as ut
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

    def prepare_data(self):
        pass

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

    def generate_datasets_rev(self, hyperparameter_test=False, hyperparameter_test_mult=0.2, 
                        short_only=False):
        size = len(self.data)
        split = (size * self.percentage) // 100

        self.test_idx = []

        self.train = []
        self.test = []

        for idx, datum in enumerate(self.data):
            if (idx < split and datum.block.num_instrs() >= 50) or (idx >= split and datum.block.num_instrs() < 50):
                self.test.append(datum)
                self.test_idx.append(idx)
            else:
                self.train.append(datum)

        if hyperparameter_test:
            self.train = self.train[:int(len(self.train) * hyperparameter_test_mult)]
            self.test = self.test[:int(len(self.test) * hyperparameter_test_mult)]
        print ('train ' + str(len(self.train)) + ' test ' + str(len(self.test)))

    def generate_batch(self, batch_size, partition=None):
        if partition is None:
            partition = (0, len(self.train))

        # TODO: this seems like it would be expensive for a large data set
        (start, end) = partition
        population = range(start, end)
        selected = random.sample(population,batch_size)

        self.batch = []
        for index in selected:
            self.batch.append(self.train[index])

    def plot_histogram(self, data):

        ys = list()
        for item in data:
          ys.append(item.y)

        plt.hist(ys, min(max(ys), 1000))
        plt.show()
