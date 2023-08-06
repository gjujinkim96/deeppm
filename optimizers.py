import torch.optim as optim

def load_optimizer(model, train_cfg):
    if train_cfg.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=train_cfg.lr)
    elif train_cfg.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=0.9, nesterov=False)
    else:
        raise NotImplementedError()
    