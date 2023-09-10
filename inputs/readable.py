from typing import Any
from .encoding import reg_dict, op_dict

def custom_token_to_readable():
    token_to_readable = {}
    reg_offset = 0
    for name, token_idx in reg_dict.items():
        token_to_readable[reg_offset + token_idx] = name
        
    int_immed_offset = reg_offset + reg_dict['REG_YMM15'] + 1
    int_immeds = ['INT_IMMED', 'INT_8_IMMED', 'INT_16_IMMED', 'INT_32_IMMED', 'INT_64_IMMED']
    for i, name in enumerate(int_immeds):
        token_to_readable[int_immed_offset + i] = name

    float_immed_offset = int_immed_offset + 4 + 1
    float_immeds = ['FLOAT_IMMED', 'FLOAT_32_IMMED', 'FLOAT_64_IMMED']
    for i, name in enumerate(float_immeds):
        token_to_readable[float_immed_offset + i] = name

    scale_offset = float_immed_offset + 2 + 1
    scales = ['SCALE_1', 'SCALE_2', 'SCALE_4', 'SCALE_8', 'SCALE_OTHER']
    for i, name in enumerate(scales):
        token_to_readable[scale_offset + i] = name

    op_codes_offset = scale_offset + 4 + 1
    for token_idx, op_name in op_dict.items():
        token_to_readable[op_codes_offset + token_idx] = op_name
    return token_to_readable

def token_to_readable():
    token_to_readable = {}
    reg_offset = 0
    for name, token_idx in reg_dict.items():
        token_to_readable[reg_offset + token_idx] = name
        
    int_immed_offset = reg_offset + reg_dict['REG_YMM15'] + 1
    token_to_readable[int_immed_offset] = 'INT_IMMED'

    float_immed_offset = int_immed_offset + 1
    token_to_readable[float_immed_offset] = 'FLOAT_IMMED'

    op_codes_offset = float_immed_offset + 1
    for token_idx, op_name in op_dict.items():
        token_to_readable[op_codes_offset + token_idx] = op_name
    return token_to_readable

class Translator:
    def __init__(self, hot_idx_to_token_mapping, token_to_readable_type='default'):
        self.hot_idx_to_token_mapping = hot_idx_to_token_mapping

        if token_to_readable_type == 'custom':
            self.token_to_readable_mapping = custom_token_to_readable()
        elif token_to_readable_type == 'default':
            self.token_to_readable_mapping = token_to_readable()
        else:
            raise NotImplementedError()

    def __call__(self, instr):
        translated = self.raw_translate(instr)
        try:
            _ = iter(instr[0])
        except TypeError:
            return ' '.join(translated)
        else:
            return '\n'.join(' '.join(line) for line in translated)
        
    def raw_translate(self, instr):
        try:
            _ = iter(instr[0])
        except TypeError:
            return self.tokens_to_readable(self.hot_idx_instr_to_tokens(instr))
        else:
            return [self.tokens_to_readable(self.hot_idx_instr_to_tokens(one_instr)) for one_instr in instr]

    def hot_idx_instr_to_tokens(self, instr):
        return [self.hot_idx_to_token_mapping[x] for x in instr]


    def tokens_to_readable(self, tokens):
        ret = []
        for token in tokens:
            if isinstance(token, int):
                ret.append(self.token_to_readable_mapping[token])
            else:
                ret.append(token)
        return ret
    