import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):  # 继承nn.Module
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3通道的输入，16卷积核，5x5的卷积核
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 用的CIFAR10数据集

    def forward(self, x):  # 正向传播的过程
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)    #View函数展平操作，-1代表纬度，自动推理纬度，32*5*5展平后节点个数
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)   #最后一层，这里不需要添加softmax层了，train.py在卷网络中，在卷积交叉熵中nn.CrossEntropyLoss()，他的内部实现了分类的功能
        return x
