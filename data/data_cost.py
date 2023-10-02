import torch

from tqdm.auto import tqdm
from .data import Data
import xml.etree.ElementTree as ET
import itertools

import sys
sys.path.append('..')

import data.utilities as ut
from collections import defaultdict
from .data_item import DataItem

from .custom_tokenizer import Tokenizer


class DataInstructionEmbedding(Data):
    def __init__(self, special_tokens=None, given_token_mapping=None):
        super(DataInstructionEmbedding, self).__init__()
        if given_token_mapping is not None:
            self.token_to_hot_idx = given_token_mapping
            self.hot_idx_to_token = {
                v: k for k, v in self.token_to_hot_idx.items()
            }
            self.next_hot_idx = max(self.token_to_hot_idx.values()) + 1
        else:
            self.token_to_hot_idx = {}
            self.hot_idx_to_token = {}
            self.next_hot_idx = 0
        self.data = []
        self.raw = []
        self.unk_tok = '<UNK>'
        self.tokenizer = None

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

    def prepare_stacked_data(self, progress=True, instr_limit=400):
        
        self.pad_idx = self.hot_idxify('<PAD>')

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            if code_intel is None or len(code_intel.split('\n')) > instr_limit:
                continue
        
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
            if len(raw_instrs) > instr_limit:
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()

            datum = DataItem(raw_instrs, timing, block, code_id)
            self.data.append(datum)

            self.raw.append(readable_instrs)

    def set_tokenizer(self, mapping):
        self.tokenizer = Tokenizer(dict(mapping))

    def prepare_stacked_raw(self, progress=True, instr_limit=400):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_raw(self.raw_data)

        self.pad_idx = self.tokenizer.mapping[self.tokenizer.pad_token]


        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            if code_intel is None or len(code_intel.split('\n')) > instr_limit:
                continue
        
            new_cur = self.tokenizer(code_intel, True, True)

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
            if len(raw_instrs) > instr_limit:
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()

            raw_instrs = new_cur

            datum = DataItem(raw_instrs, timing, block, code_id)
            self.data.append(datum)

            self.raw.append(readable_instrs)

        self.token_to_hot_idx = dict(self.tokenizer.mapping)
        self.hot_idx_to_token = dict(self.tokenizer.rev_mapping)

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

def load_data(data_savefile, small_size=False, only_unique=False,
            split_mode='none', split_perc=(8, 2, 0),
            special_tokens=None, 
            prepare_mode='stacked', shuffle=False, given_token_mapping=None,
            instr_limit=400, given_train_val_test_idx=None,
    ):
    '''
    split_mode: num_instrs, none
    prepare_mode: stacked, stacked_raw
    '''
    data = DataInstructionEmbedding(special_tokens=special_tokens, given_token_mapping=given_token_mapping)

    if small_size:
        data.raw_data = torch.load(data_savefile)[:100]
    else:
        data.raw_data = torch.load(data_savefile)
    data.read_meta_data()

    if prepare_mode == 'stacked':
        data.prepare_stacked_data(instr_limit=instr_limit)
    elif prepare_mode == 'stacked_raw':
        if given_token_mapping is not None:
            data.set_tokenizer(given_token_mapping)
        data.prepare_stacked_raw(instr_limit=instr_limit)
    else:
        raise NotImplementedError()

    if only_unique:
        data.data, data.raw = extract_unique(data.data, data.raw)

    # change test pert or some way to split train, val test in constructor? no in generate dataset!!
    #  remove self.perct in base class
    data.generate_datasets(split_mode=split_mode, split_perc=split_perc,
                        shuffle=shuffle, given_train_val_test_idx=given_train_val_test_idx, small_size=small_size
    )

    return data
