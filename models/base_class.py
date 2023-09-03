import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = get_device(should_print=False)

    def run_train(self, input, loss_mod=None):
        x = input['x'].to(self.device)
        output = self.forward(x)
        loss_fn = self.get_loss()

        y = input['y'].to(self.device)
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        return {'loss': loss.item()}, y.tolist(), output.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x'].to(self.device)
        output = self.forward(x)
        loss_fn = self.get_loss()

        y = input['y'].to(self.device)
        loss = loss_fn(output, y)
        
        if loss_mod is not None:
            loss *= loss_mod

        return {'loss': loss.item()}, y.tolist(), output.tolist()
    
    def run(self, input, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(input, loss_mod=loss_mod)
        else:
            return self.run_val(input, loss_mod=loss_mod)

    def get_loss(self):
        return self.loss


class CheckpointModule(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.device = get_device(should_print=False)

    def checkpoint_forward(self, x):
        raise NotImplementedError()
    
    def run_train(self, input, loss_mod=None):
        x = input['x'].to(self.device)

        if self.use_checkpoint:
            output = self.checkpoint_forward(x)
        else:
            output = self.forward(x)

        loss_fn = self.get_loss()
        y = input['y'].to(self.device)
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        return {'loss': loss.item()}, y.tolist(), output.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x'].to(self.device)
        if self.use_checkpoint:
            output = self.checkpoint_forward(x)
        else:
            output = self.forward(x)
            
        loss_fn = self.get_loss()
        y = input['y'].to(self.device)
        loss = loss_fn(output, y)
        
        if loss_mod is not None:
            loss *= loss_mod

        return {'loss': loss.item()}, y.tolist(), output.tolist()
    
    def run(self, input, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(input, loss_mod=loss_mod)
        else:
            return self.run_val(input, loss_mod=loss_mod)
    
    def get_loss(self):
        return self.loss
    
class BertModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = get_device(should_print=False)

    def run_train(self, input, loss_mod=None):
        x = input['x'].to(self.device)
        masked_x = input['masked_x'].to(self.device)

        prediction = self.forward(masked_x)

        batch_size, inst_size, seq_size = x.shape
        x = x.view(batch_size * inst_size *seq_size)
        prediction = prediction.view(batch_size * inst_size *seq_size, -1)
        loss_fn = self.get_loss()
        loss = loss_fn(prediction, x)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        prediction = prediction.detach()
        correct = (x == torch.max(prediction, dim=-1)[1]).sum().item()

        total = (x != 0).sum().item()

        return {'loss': loss.item()}, correct, total

    def run_val(self, input, loss_mod=None):
        x = input['x'].to(self.device)
        masked_x = input['masked_x'].to(self.device)

        prediction = self.forward(masked_x)

        batch_size, inst_size, seq_size = x.shape
        x = x.view(batch_size * inst_size *seq_size)
        prediction = prediction.view(batch_size * inst_size *seq_size, -1)
        loss_fn = self.get_loss()
        loss = loss_fn(prediction, x)

        if loss_mod is not None:
            loss *= loss_mod

        prediction = prediction.detach()
        correct = (x == torch.max(prediction, dim=-1)[1]).sum().item()
        total = (x != 0).sum().item()

        return {'loss': loss.item()}, correct, total
    
    def run(self, input, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(input, loss_mod=loss_mod)
        else:
            return self.run_val(input, loss_mod=loss_mod)

    def get_loss(self):
        return self.loss
    
# class BertCheckpointModule(nn.Module):
#     def __init__(self, use_checkpoint=False):
#         super().__init__()

#         self.use_checkpoint = use_checkpoint
#         self.device = get_device(should_print=False)
    
#     def checkpoint_forward(self, x):
#         raise NotImplementedError()
    
#     def run_train(self, input, loss_mod=None):
#         x = input['x'].to(self.device)

#         if self.use_checkpoint:
#             output = self.checkpoint_forward(x)
#         else:
#             output = self.forward(x)

#         loss_fn = self.get_loss()
#         y = input['y'].to(self.device)
#         loss = loss_fn(output, y)

#         if loss_mod is not None:
#             loss *= loss_mod
#         loss.backward()

#         return {'loss': loss.item()}, y.tolist(), output.tolist()

#     def run_train(self, input, loss_mod=None):
#         x = input['x'].to(self.device)
#         masked_x = input['masked_x'].to(self.device)

#         prediction = self.forward(masked_x)

#         batch_size, inst_size, seq_size = x.shape
#         x = x.view(batch_size * inst_size *seq_size)
#         prediction = prediction.view(batch_size * inst_size *seq_size, -1)
#         loss_fn = self.get_loss()
#         loss = loss_fn(prediction, x)

#         if loss_mod is not None:
#             loss *= loss_mod
#         loss.backward()

#         prediction = prediction.detach()
#         correct = (x == torch.max(prediction, dim=-1)[1]).sum().item()

#         total = (x != 0).sum().item()

#         return {'loss': loss.item()}, correct, total
    
#         x = input['x'].to(self.device)
#         masked_x = input['masked_x'].to(self.device)

#         prediction = self.forward(masked_x)

#         batch_size, inst_size, seq_size = x.shape
#         x = x.view(batch_size * inst_size *seq_size)
#         prediction = prediction.view(batch_size * inst_size *seq_size, -1)
#         loss_fn = self.get_loss()
#         loss = loss_fn(prediction, x)

#         if loss_mod is not None:
#             loss *= loss_mod
#         loss.backward()

#         prediction = prediction.detach()
#         correct = (x == torch.max(prediction, dim=-1)[1]).sum().item()

#         total = (x != 0).sum().item()

#         return {'loss': loss.item()}, correct, total

#     def run_val(self, input, loss_mod=None):
#         x = input['x'].to(self.device)
#         masked_x = input['masked_x'].to(self.device)

#         prediction = self.forward(masked_x)

#         batch_size, inst_size, seq_size = x.shape
#         x = x.view(batch_size * inst_size *seq_size)
#         prediction = prediction.view(batch_size * inst_size *seq_size, -1)
#         loss_fn = self.get_loss()
#         loss = loss_fn(prediction, x)

#         if loss_mod is not None:
#             loss *= loss_mod

#         prediction = prediction.detach()
#         correct = (x == torch.max(prediction, dim=-1)[1]).sum().item()
#         total = (x != 0).sum().item()

#         return {'loss': loss.item()}, correct, total
    
#     def run(self, input, loss_mod=None, is_train=False):
#         if is_train:
#             return self.run_train(input, loss_mod=loss_mod)
#         else:
#             return self.run_val(input, loss_mod=loss_mod)

#     def get_loss(self):
#         raise NotImplementedError()

class UnRollingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def run_train(self, input, loss_mod=None):
        x = input['x']
        y = input['y']
        unrolled = input['unrolled']
        inst_lens = torch.tensor(input['inst_len'])
        
        output_sum, outputs = self.forward(x, unrolled)
        loss_fn, unrolled_loss_fn = self.get_loss()


        loss_dict = {}
        
        if unrolled.any():
            unrolled_outputs = outputs[unrolled]
            unrolled_inst_lens = inst_lens[unrolled]
            half = torch.div(unrolled_inst_lens, 2, rounding_mode='trunc')

            f = []
            s = []
            for idx, t in enumerate(unrolled_outputs):
                for i in range(half[idx]):
                    f.append(t[i])
                    s.append(t[i+half[idx]])
            f = torch.stack(f)
            s = torch.stack(s)
            unrolled_loss = unrolled_loss_fn(f, s)

            if loss_mod is not None:
                unrolled_loss *= loss_mod

            unrolled_loss.backward(retain_graph=True)
            loss_dict['unrolled'] = unrolled_loss.item()
        
        loss = loss_fn(output_sum, y)
        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        loss_dict['loss'] = loss.item()

        return loss_dict, y.tolist(), output_sum.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x']
        y = input['y']
        unrolled = input['unrolled']
        inst_lens = torch.tensor(input['inst_len'])
        
        output_sum, outputs = self.forward(x, unrolled)
        loss_fn, unrolled_loss_fn = self.get_loss()


        loss_dict = {}
        
        if unrolled.any():
            unrolled_outputs = outputs[unrolled]
            unrolled_inst_lens = inst_lens[unrolled]
            half = torch.div(unrolled_inst_lens, 2, rounding_mode='trunc')

            f = []
            s = []
            for idx, t in enumerate(unrolled_outputs):
                for i in range(half[idx]):
                    f.append(t[i])
                    s.append(t[i+half[idx]])
            f = torch.stack(f)
            s = torch.stack(s)
            unrolled_loss = unrolled_loss_fn(f, s)

            if loss_mod is not None:
                unrolled_loss *= loss_mod

            loss_dict['unrolled'] = unrolled_loss.item()
        
        loss = loss_fn(output_sum, y)
        if loss_mod is not None:
            loss *= loss_mod

        loss_dict['loss'] = loss.item()

        return loss_dict, y.tolist(), output_sum.tolist()
    
    def run(self, input, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(input, loss_mod=loss_mod)
        else:
            return self.run_val(input, loss_mod=loss_mod)

    def get_loss(self):
        raise NotImplementedError()
    