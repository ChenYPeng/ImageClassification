import torch

def get_criterion(criterion_name, weight):
    if criterion_name == 'cross':
       criterion = torch.nn.CrossEntropyLoss(weight=None)
    else:
        raise AttributeError("Unsupported criterion type: {}".format(criterion_name))
    return criterion