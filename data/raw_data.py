import torch
import numpy as np

import data.utilities as ut

class MetaData:
    def __init__(self):
        self.costs = dict()

    def read_data(self):
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
            self.costs[i] = np.random.randint(1, maxnum)


class RawData:
    def __init__(self, data_savefile, small_size=False, use_metadata=False):
        self.small_size = small_size

        self.data = torch.load(data_savefile)

        if self.small_size:
            self.data = self.data[:100]

        if use_metadata:
            self.meta = MetaData()
            self.meta.read_data()
        else:
            self.meta = None
        
