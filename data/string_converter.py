from .custom_tokenizer import Tokenizer
from tqdm.auto import tqdm

class StringDataItem:
    def __init__(self, x, y, code_id, num_instrs, raw):
        self.x = x
        self.y = y
        self.code_id = code_id
        self.num_instrs = num_instrs
        self.raw = raw

    def __repr__(self):
        return f'id:\t{self.code_id}\ny:\t{self.y}\nbasic block\n{self.raw}'

    def __str__(self):
        return self.__repr__()

class StringConverter:
    def __init__(self, special_tokens=None, given_token_mapping=None):
        self.unk_tok = '<UNK>'
        self.pad_tok = '<PAD>'
        self.special_tokens = {}

        if given_token_mapping is not None:
            self.tokenizer = Tokenizer(dict(given_token_mapping))
        else:
            self.tokenizer = None
            if special_tokens is not None:
                self.next_hot_idx = max(special_tokens.values()) + 1
                for k, v in special_tokens.items():
                    splitted = k.split('_')
                    token = f'<{k}>'
                    self.special_tokens[token] = v

    def dump_params(self):
        return self.tokenizer.mapping
   
    
    def convert(self, raw_data, progress=True, instr_limit=400):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_raw(raw_data.data, special_toks=self.special_tokens)
        converted_data = []

        if progress:
            iterator = tqdm(raw_data.data)
        else:
            iterator = raw_data.data

        for (code_id, timing, code_intel, code_xml) in iterator:
            if code_intel is None or len(code_intel.split('\n')) > instr_limit:
                continue

            tok_basic_block = self.tokenizer(code_intel, True, True)
          
            bb_len = len(tok_basic_block)
            if bb_len > instr_limit:
                continue

            datum = StringDataItem(tok_basic_block, timing, code_id, bb_len, code_intel)

            converted_data.append(datum)
        
        return converted_data
    