import torch

def get_scheduler(scheduler_name, optimizer, step_size=5):
    if scheduler_name == 'step':
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.65)
    else:
        raise AttributeError("Unsupported scheduler type: {}".format(scheduler_name))
    return scheduler