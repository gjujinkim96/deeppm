import torch
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def run_train(self, input, loss_mod=None):
        x = input['x']
        y = input['y']
        output = self.forward(x)
        loss_fn = self.get_loss()
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        return {'loss': loss.item()}, y.tolist(), output.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x']
        y = input['y']
        output = self.forward(x)
        loss_fn = self.get_loss()
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
        raise NotImplementedError()


class CheckpointModule(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

    def checkpoint_forward(self, x):
        raise NotImplementedError()
    
    def run_train(self, input, loss_mod=None):
        x = input['x']
        y = input['y']

        if self.use_checkpoint:
            output = self.checkpoint_forward(x)
        else:
            output = self.forward(x)

        loss_fn = self.get_loss()
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod
        loss.backward()

        return {'loss': loss.item()}, y.tolist(), output.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x']
        y = input['y']
        if self.use_checkpoint:
            output = self.checkpoint_forward(x)
        else:
            output = self.forward(x)
            
        loss_fn = self.get_loss()
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
        raise NotImplementedError()
    

class BertModule(nn.Module):
    def __init__(self):
        super().__init__()

    def run_train(self, input, loss_mod=None):
        x = input['x']
        y = input['y']

        output, bert_output = self.forward(x)
        loss_fn, bert_loss_fn = self.get_loss()
        loss = loss_fn(output, y)

        batch_size, inst_size, seq_size = x.shape
        bert_target = x.view(-1)
        bert_output = bert_output.view(batch_size * inst_size * seq_size, -1)
        bert_loss = bert_loss_fn(bert_output, bert_target)

        if loss_mod is not None:
            loss *= loss_mod
            bert_loss *= loss_mod
        bert_loss.backward(retain_graph=True)
        loss.backward()

        return {'loss': loss.item(), 'bert_loss': bert_loss.item()}, y.tolist(), output.tolist()

    def run_val(self, input, loss_mod=None):
        x = input['x']
        y = input['y']

        output, bert_output = self.forward(x)
        loss_fn, bert_loss_fn = self.get_loss()
        loss = loss_fn(output, y)

        batch_size, inst_size, seq_size = x.shape
        bert_target = x.view(-1)
        bert_output = bert_output.view(batch_size * inst_size * seq_size, -1)
        bert_loss = bert_loss_fn(bert_output, bert_target)
        
        if loss_mod is not None:
            loss *= loss_mod
            bert_loss *= loss_mod

        return {'loss': loss.item(), 'bert_loss': bert_loss.item()}, y.tolist(), output.tolist()
    
    def run(self, input, loss_mod=None, is_train=False):
        if is_train:
            return self.run_train(input, loss_mod=loss_mod)
        else:
            return self.run_val(input, loss_mod=loss_mod)

    def get_loss(self):
        raise NotImplementedError()

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
    