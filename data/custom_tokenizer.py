import re
from tqdm.auto import tqdm

class Tokenizer:
    pad_token = '<PAD>'
    block_start_token = '<BLOCK_START>'
    block_end_token = '<BLOCK_END>'
    start_token = '<START>'
    end_token = '<END>'
    sep_token = '<SEP>'
    unk_token = '<UNK>'
    msk_token = '<MSK>'
    
    padding_tokens = '[ ] + - * , :'.split()
    replace_tokens = {
        '\n': '<SEP>',
        '<INVALID>': unk_token,
    }

    
    def __init__(self, tokenizer_mapping):
        self.mapping = tokenizer_mapping
        self.rev_mapping = {idx: token for token, idx in self.mapping.items()}
        
    def __call__(self, line, postprocess=True, stackify=False):
        ret = self.normalize(line)
        ret = self.pretokenize(ret)

        if stackify:
            ret = self.stackify(ret)
            if postprocess:
                stack_size = len(ret)
                ret = [self.postprocess(line, idx == 0, idx == stack_size - 1) for idx, line in enumerate(ret)]

            ret = [self.indexify(line) for line in ret]
        else:
            if postprocess:
                ret = self.postprocess(ret)
            ret = self.indexify(ret)    
        
        return ret
        

    @classmethod
    def normalize(cls, line):
        mod_line = line
        for replace_src, replace_dst in cls.replace_tokens.items():
            mod_line = mod_line.replace(replace_src, f' {replace_dst} ')
            
        for pad_token in cls.padding_tokens:
            mod_line = mod_line.replace(pad_token, f' {pad_token} ')
    
        mod_line = re.sub(r'\b0x(0+)\b', lambda x: f'<ZERO_{len(x.group(1))//2}_BYTES>', mod_line)
        mod_line = re.sub(r'\b0x([0-9a-f]+)\b', lambda x: f'<NUM_{len(x.group(1))//2}_BYTES>', mod_line)
        mod_line = ' '.join(mod_line.split())
        return mod_line

    @classmethod
    def pretokenize(cls, norm_line):
        return norm_line.split()

    @classmethod
    def postprocess(cls, tok_list, stack_first=False, stack_end=False):
        start_token = cls.block_start_token if stack_first else cls.start_token
        end_token = cls.block_end_token if stack_end else cls.end_token
        return [start_token] + tok_list + [end_token]

    @classmethod
    def stackify(cls, tok_list):
        ret = []
        cur = []
        for tok in tok_list:
            if tok == cls.sep_token and len(cur) > 0:
                ret.append(cur)
                cur = []
            else:
                cur.append(tok)
        if len(cur) > 0:
            ret.append(cur)
        return ret

    def indexify(self, tok_list):
        ret = []
        for tok in tok_list:
            if tok not in self.mapping:
                tok = self.unk_token
            ret.append(self.mapping[tok])
        return ret

    def tokenify(self, idx_list):
        if isinstance(idx_list[0], list):
            return [[self.rev_mapping[idx] for idx in line] for line in idx_list]
        else:
            return [self.rev_mapping[idx] for idx in idx_list]

    def stringify(self, idx_list):
        ret = self.tokenify(idx_list)
        if isinstance(idx_list[0], list):
            return '\n'.join([' '.join(line) for line in ret])
        else:
            return ' '.join(ret)


    @classmethod
    def from_raw(cls, raw):
        special_toks = [cls.pad_token, 
            cls.block_start_token, cls.block_end_token,
            cls.start_token, cls.end_token, cls.sep_token, cls.unk_token, cls.msk_token]
        tokenizer_mapping = {tok: idx for idx, tok in enumerate(special_toks)}
        
        lines = [line[2] for line in raw]  
        for line in tqdm(lines):
            mod_line = cls.normalize(line)
            tokens = cls.pretokenize(mod_line)
            for token in tokens:
                if token not in tokenizer_mapping:
                    tokenizer_mapping[token] = len(tokenizer_mapping)

        return cls(tokenizer_mapping)
     