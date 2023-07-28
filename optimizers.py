import torch.optim as optim

def load_optimizer(model, train_cfg):
    if train_cfg.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=train_cfg.lr)
    else:
        raise NotImplementedError()
    