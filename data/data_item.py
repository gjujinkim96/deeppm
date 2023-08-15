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
