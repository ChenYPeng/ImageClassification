from datasets.pokemon import Pokemon


def get_dataset(data_path):
    train_set = Pokemon(root=data_path, size=256, split='train')
    val_set = Pokemon(root=data_path, size=256, split='val')
    test_set = Pokemon(root=data_path, size=256, split='test')
    return train_set, val_set, test_set