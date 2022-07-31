import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from net import get_model
from config import get_config
from mypath import get_datapath
from datasets import get_dataset

def denormalize_image(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    image *= std
    image += mean
    image *= 255

    return image

if __name__ == '__main__':

    config_name = 'pokemon.yaml'
    config = get_config(config_name)
    params = config['parameters']
    dataset = params['dataset']
    num_classes = params['num_classes']
    model_name= params['model_name']

    # 指定训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取数据集
    data_path = get_datapath(dataset)
    _, _, test_set = get_dataset(data_path)

    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    # 测试

    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('checkpoint/best_model.pth'))

    class_names = ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']

    correct=0.0
    total=0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(test_loader)): 
            inputs, targets = inputs.to(device), targets.to(device)
            # 输出特征预测值
            outputs = model(inputs)
            preds = outputs.argmax(1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            plt.figure(figsize=(12, 8))
            for i in range(inputs.size(0)):
                plt.subplot(4, 4, i + 1)  #根据 batch_size修改
                plt.axis('off')
                plt.title(f'{class_names[preds[i]]} || {class_names[targets[i]]}')
                img = np.transpose(inputs[i].cpu().numpy(), (1, 2, 0))
                img = denormalize_image(img).astype(np.uint8)
                plt.imshow(img)
            plt.savefig('vis/batch_idx_{}.jpg'.format(batch_idx), bbox_inches='tight')
        print("Test Average accuracy is:{:.4f}%".format(100 * correct / total))
