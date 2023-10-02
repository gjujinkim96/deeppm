import torch.optim.lr_scheduler as lr_sched


def decay_after_delay(delay=2, div_factor=1.2):
    def lr_sched(step):
        if step < delay:
            return 1
        return 1/1.2**(step-delay+1)
    return lr_sched

def get_decay_after_delay_lr_sched(opt, delay=2, div_factor=1.2):
    return lr_sched.LambdaLR(opt, decay_after_delay(delay=delay, div_factor=div_factor))
