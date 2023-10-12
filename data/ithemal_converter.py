import data.utilities as ut
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import itertools

class IthemalDataItem:
    def __init__(self, x, y, block, code_id, raw):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id
        self.num_instrs = self.block.num_instrs()
        self.raw = raw

class IthemalConverter:
    def __init__(self, special_tokens=None, given_token_mapping=None):
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
            self.unk_tok = '<UNK>'
            self.pad_tok = '<PAD>'

            if special_tokens is not None:
                self.next_hot_idx = max(special_tokens.values()) + 1
                for k, v in special_tokens.items():
                    splitted = k.split('_')
                    token = f'<{k}>'
                    self.token_to_hot_idx[token] = v
                    self.hot_idx_to_token[v] = token

            self.hot_idxify(self.unk_tok)
            self.pad_idx = self.hot_idxify(self.pad_tok)

    def dump_params(self):
        return self.token_to_hot_idx
    
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
    
    def convert(self, raw_data, progress=True, instr_limit=400):
        converted_data = []

        if progress:
            iterator = tqdm(raw_data.data)
        else:
            iterator = raw_data.data

        for (code_id, timing, code_intel, code_xml) in iterator:
            if code_intel is None or len(code_intel.split('\n')) > instr_limit:
                continue

            block_root = ET.fromstring(code_xml)
            instrs = []

            raw_instrs = []
            curr_mem = raw_data.meta.mem_start
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

                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel
            if len(raw_instrs) > instr_limit:
                continue

            block = ut.BasicBlock(instrs)
            block.create_dependencies()

            datum = IthemalDataItem(raw_instrs, timing, block, code_id, code_intel)
            converted_data.append(datum)
        
        return converted_data
    