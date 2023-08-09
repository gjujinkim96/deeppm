import torch
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def run_train(self, x, y, loss_mod=None):
        output = self.forward(x)
        loss_fn = self.get_loss()
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        return loss.item(), y.tolist(), output.tolist()

    def run_val(self, x, y, loss_mod=None):
        output = self.forward(x)
        loss_fn = self.get_loss()
        loss = loss_fn(output, y)
        
        if loss_mod is not None:
            loss *= loss_mod

        return loss.item(), y.tolist(), output.tolist()
    
    def run(self, x, y, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(x, y, loss_mod=loss_mod)
        else:
            return self.run_val(x, y, loss_mod=loss_mod)

    def get_loss(self):
        raise NotImplementedError()
