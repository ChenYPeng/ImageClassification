from net.dendenet import DenseNet



def get_model(model_name, num_classes):

    if model_name == 'densenet':
        model = DenseNet(num_classes=num_classes)
    
    return model
