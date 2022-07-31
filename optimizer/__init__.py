import torch

def get_optimizer(optimizer_name, model, lr, momentum):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model, lr=lr, momentum=momentum)
    else:
        raise AttributeError("Unsupported optimizer type: {}".format(optimizer_name))
    return optimizer