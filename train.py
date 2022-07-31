import torch
import tqdm 
from torch.utils.data import DataLoader

from net import get_model
from mypath import get_datapath
from datasets import get_dataset
from config import get_config
from optimizer import get_optimizer
from criterion import get_criterion
from scheduler import get_scheduler

def train(model, epoch, train_loader, criterion, optimizer, scheduler, max_epoch):
    running_loss = 0.0
    running_correct=0.0

    model.train()
    with tqdm.tqdm(train_loader) as train_bar:
        train_bar.set_description('Train Epoch[{:3d}/{:3d}]'.format(epoch, max_epoch))
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            # 输出特征预测值
            outputs = model(data)
            # 计算损失
            loss = criterion(outputs, target)

            # 优化器梯度清 0
            optimizer.zero_grad()

            # 计算梯度
            loss.backward()

            # 更新梯度
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            running_correct += torch.sum(pred == target.data)
            train_bar.set_postfix({'train_batch_loss': '{0:1.5f}'.format(loss.item())})
        scheduler.step()
        train_bar.update(len(train_loader))
        print("Train Average accuracy is:{:.4f}%".format(100 * running_correct / len(train_set)))

def valid(model, epoch, val_loader, max_epoch):
    model.eval()
    best_acc = 0.0
    correct=0.0
    total=0.0

    with tqdm.tqdm(val_loader) as eval_bar:
        eval_bar.set_description('Valid Epoch[{:3d}/{:3d}]'.format(epoch, max_epoch))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(eval_bar): 
                data, target = data.to(device), target.to(device)
                # 输出特征预测值
                outputs = model(data)

                loss = criterion(outputs, target)

                _, predicted = torch.max(outputs.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

                eval_bar.set_postfix({'valid_batch_loss': '{0:1.5f}'.format(loss.item())})
            test_acc = correct / total
            
            # 保存模型
            new_acc = test_acc
            if test_acc > best_acc:
                best_acc = new_acc
                torch.save(model.state_dict(), "checkpoint/best_model.pth")
        eval_bar.update(len(val_loader))
        print("Test Average accuracy is:{:.4f}%".format(100 * test_acc))
    

if __name__ == '__main__':

    # 获取配置参数
    config_name = 'pokemon.yaml'
    config = get_config(config_name)
    params = config['parameters']
    batch_size=params['batch_size']
    epochs=params['epochs']
    lr=params['lr']
    momentum = params['momentum']
    num_classes = params['num_classes']
    model_name= params['model_name']
    dataset = params['dataset']
    optimizer_name = params['optimizer_name']
    criterion_name = params['criterion_name']
    scheduler_name = params['scheduler_name']

    # 指定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name=model_name, num_classes=num_classes).to(device)

    criterion = get_criterion(criterion_name=criterion_name, weight=None)
    optimizer = get_optimizer(optimizer_name=optimizer_name, model=model.parameters(), lr=lr, momentum=momentum)
    scheduler = get_scheduler(scheduler_name=scheduler_name, optimizer=optimizer)

    # 获取数据集
    data_path = get_datapath(dataset)
    train_set, val_set, _ = get_dataset(data_path)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 训练
    for epoch in range(epochs):
        train(model, epoch, train_loader, criterion, optimizer, scheduler, epochs)
        valid(model, epoch, val_loader, epochs)

        
