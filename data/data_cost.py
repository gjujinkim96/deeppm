import numpy as np
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .data import Data
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import xml.etree.ElementTree as ET
import itertools

import os
import sys
sys.path.append('..')

import data.utilities as ut
from collections import defaultdict


class DataItem:
    def __init__(self, x, y, block, code_id, src=None):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id
        self.src = src

    def __repr__(self):
        return f'---- Block ----\n{self.block}\nX: {self.x}  Y: {self.y}'

class DataItemWithSim:
    def __init__(self, x, y, block, code_id, sim):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id
        self.sim = sim

    def __repr__(self):
        return f'---- Block ----\n{self.block}\nX: {self.x}  Y: {self.y}'

class DataInstructionEmbedding(Data):
    def __init__(self, special_tokens=None):
        super(DataInstructionEmbedding, self).__init__()
        self.token_to_hot_idx = {}
        self.hot_idx_to_token = {}
        self.data = []
        self.raw = []
        self.unk_tok = '<UNK>'
        self.next_hot_idx = 0

        if special_tokens is not None:
            self.next_hot_idx = max(special_tokens.values()) + 1
            for k, v in special_tokens.items():
                splitted = k.split('_')
                if len(splitted) > 1 and splitted[1] == 'FIN':
                    token = f'</{k}>'
                else:
                    token = f'<{k}>'
                self.token_to_hot_idx[token] = v
                self.hot_idx_to_token[v] = token

    def dump_dataset_params(self):
        return (self.token_to_hot_idx, self.hot_idx_to_token)

    def load_dataset_params(self, params):
        (self.token_to_hot_idx, self.hot_idx_to_token) = params

    def hot_idxify(self, elem, fixed=False):
        if elem not in self.token_to_hot_idx:
            if fixed:
                if self.unk_tok not in self.token_to_hot_idx:
                    self.token_to_hot_idx[self.unk_tok] = self.next_hot_idx
                    self.next_hot_idx += 1
                    self.hot_idx_to_token[self.token_to_hot_idx[self.unk_tok]] = self.unk_tok
                return self.token_to_hot_idx[self.unk_tok]
            else:
                self.token_to_hot_idx[elem] = self.next_hot_idx
                self.next_hot_idx += 1
                self.hot_idx_to_token[self.token_to_hot_idx[elem]] = elem
        return self.token_to_hot_idx[elem]

    def prepare_data(self, progress=True):
        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            
            #if timing > 1112:#1000:
            #    continue

            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            readable_raw = ['<START>']
            curr_mem = self.mem_start
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    #raw_instr.append(opcode)
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    readable_raw.extend(raw_instr)
                    # raw_instrs.extend(list(map(hot_idxify, raw_instr)))
                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel

            readable_raw.append('<DONE>')
            raw_instrs = list(map(self.hot_idxify, readable_raw))
            if len(raw_instrs) > 4000:
                #print(len(raw_instrs))
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id)

 
            self.data.append(datum)
            self.raw.append(readable_raw)


    def prepare_stacked_data(self, progress=True, src_df=None):
        
        self.pad_idx = self.hot_idxify('<PAD>')

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
        
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            readable_instrs = []
            curr_mem = self.mem_start
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(self.hot_idxify, raw_instr)))
                    readable_instrs.append(raw_instr)

                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel
            if len(raw_instrs) > 400:
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()

            if src_df is not None:
                src_type = src_df.loc[code_id].mapping
                datum = DataItem(raw_instrs, timing, block, code_id, src_type)
            else:
                datum = DataItem(raw_instrs, timing, block, code_id)
            self.data.append(datum)

            self.raw.append(readable_instrs)

    def prepare_simplified(self, progress=True):
        sim_mapping = {}
        def sim_idxify(token):
            if token not in sim_mapping:
                sim_mapping[token] = len(sim_mapping)
            return sim_mapping[token]
            
        self.pad_idx = self.hot_idxify('<PAD>')
        sim_idxify('<PAD>')

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            readable_instrs = []
            curr_mem = self.mem_start
            sims = []
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    src_mem_cnt = 0
                    dst_mem_cnt = 0
                    raw_instr.extend([opcode, '<SRCS>'])
                    
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                            src_mem_cnt += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                            dst_mem_cnt += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(self.hot_idxify, raw_instr)))
                    readable_instrs.append(raw_instr)
                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel
                    sims.append(sim_idxify(f'{opcode}_{src_mem_cnt}_{dst_mem_cnt}'))
            if len(raw_instrs) > 400:
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItemWithSim(raw_instrs, timing, block, code_id, sims)
            self.data.append(datum)

            self.raw.append(readable_instrs)


def extract_unique(data, raw):    
    grouped = defaultdict(list)
    for idx, datum in enumerate(tqdm(data, total=len(data))):
        grouped[str(datum.block.instrs)].append(idx)
        
    cleaned = []
    cleaned_raw = []
    for k, v in tqdm(grouped.items(), total=len(grouped)):
        vs = [data[idx].y for idx in v]
        new_y = sum(vs) / len(vs)
        datum = data[v[0]]
        datum.y = new_y
        cleaned.append(datum)
        cleaned_raw.append(raw[v[0]])
    
    print(f'{len(cleaned)} unique data found!')
    return cleaned, cleaned_raw

def load_data(data_savefile, small_size=False, stacked=False, only_unique=False, split_mode='none', split_perc=(8, 2, 0), hyperparameter_test=False,
                    hyperparameter_test_mult=0.2, special_tokens=None, simplify=False, src_info_file=None):
    '''
    split_mode: num_instrs+srcs, num_instrs, none
    '''
    data = DataInstructionEmbedding(special_tokens=special_tokens)

    if small_size:
        data.raw_data = torch.load(data_savefile)[:100]
    else:
        data.raw_data = torch.load(data_savefile)
    data.read_meta_data()

    if src_info_file is not None:
        src_df = pd.read_csv(src_info_file, index_col='idx')
    else:
        src_df = None

    if simplify:
        raise NotImplementedError()
        data.prepare_simplified()
    elif stacked:
        data.prepare_stacked_data(src_df=src_df)
    else:
        raise NotImplementedError()
        data.prepare_data()

    if only_unique:
        data.data, data.raw = extract_unique(data.data, data.raw)

    # change test pert or some way to split train, val test in constructor? no in generate dataset!!
    #  remove self.perct in base class
    data.generate_datasets(split_mode=split_mode, split_perc=split_perc, hyperparameter_test=hyperparameter_test, 
                        hyperparameter_test_mult=hyperparameter_test_mult,
                        )

    return data
