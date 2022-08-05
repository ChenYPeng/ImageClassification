from network.DenseNet import DenseNet
from network.AlexNet import AlexNet



def get_model(model_name, num_classes):

    if model_name == 'densenet':
        model = DenseNet(num_classes=num_classes)
    elif model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    else:
        raise AttributeError("Unsupported net type: {}".format(model_name))
    
    return model
