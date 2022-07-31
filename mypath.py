import os


def get_datapath(dataset):
    if dataset == 'pokemon':
        return 'E:/Datasets/pokemon/'  # folder that contains pokemon/.
    else:
        print('Dataset {} not available.'.format(dataset))
        raise NotImplementedError

