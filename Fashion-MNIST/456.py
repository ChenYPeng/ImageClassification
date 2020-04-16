import torch                            # 引入相关的包
import torchvision
import torch.nn as nn
import torch.nn.functional as F         # 引用神经网络常用函数包，不具有可学习的参数
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print(" x shape ",x.size())
        x = x.view(-1,128*8*8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x


# 定义曲线样式
def plot_curve(loss_list, label_name):
    plt.figure()  # 定义图形对象
    plt.plot(range(len(loss_list)), loss_list, color='blue')  # 关联x和y轴数据
    plt.legend([label_name], loc='upper right')  # 设置坐标轴方向
    plt.xlabel('epoch')   # 设置x，y标识符
    plt.ylabel('value')
    plt.show()


# 定义显示图片模式
def plot_image(img, label, name):
    plt.figure()
    text_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(9):
        plt.subplot(3, 3, i + 1)  # 设置为2行3列
        plt.tight_layout()  # 设置为固定布局
        plt.imshow(
            img[i][0] * 0.3081 + 0.1307,
            cmap='gray',
            interpolation='none')
        plt.title("{}: {}".format(name, text_list[label[i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 定义训练函数
def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()  # 把模型切换到train模式
    # 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
    train_loss_list = []  # 保存损失变化过程的数组
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):  # 批次，输入，标签
        images, labels = images.to(device), labels.to(device)   # 输入和标签进入显卡
        optimizer.zero_grad()  # 清空梯度，在每次优化前都要进行此操作
        output = model(images)  # 用网络计算当前输出
        loss = F.nll_loss(output, labels)  # 调用交叉熵损失函数
        loss.backward()  # 损失的反向传播
        optimizer.step()  # 根据parameter的梯度更新parameter的值
        scheduler.step()
        train_loss += loss.item()  # 增加一次损失数据
        train_loss_list.append(train_loss / len(train_loader))  # 每个epoch添加一次,平均训练误差
        if (batch_idx+1) % 100 == 0 and batch_idx != 0:
            torch.save(model, 'model_Net.pkl')
            print('saved')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    plot_curve(train_loss_list, 'Training loss')
    print('Finished Training')


# 定义测试函数
def test(model, device, test_loader):
    model.eval()  # 把模型切换到evaluation模式
    test_loss_list = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # 对测试集中的所有图片都过一遍
        for images, labels in test_loader:
            # 对传入的测试集图片进行正向推断、计算损失，total_accuracy为测试集一万张图片中模型预测正确率
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            # sum up batch loss
            test_loss += F.nll_loss(output, labels, reduction='sum').item()
            # 取该批测试数据中每个输出中，概率最大的作为该个数据的预测值
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
    test_loss_list.append(test_loss/len(test_loader))  # 增加一次损失数据
    plot_curve(test_loss_list, 'Validation loss')
    plot_image(images.cpu(), pred.cpu(), 'test')  # 从显卡回到CPU并显示其中6个
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 定义主函数
def main():
    # 定义超参数
    epoch = 3
    batch_size = 16
    lr = 0.01
    momentum = 0.9

    # 下载Fashion_MNIST训练集数据，并构建训练集数据载入器train_loader,
    train_loader = torch.utils.data.DataLoader(  # 加载训练数据集
        torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True,
                                          # 指定数据目录，类型为训练集，没有就下载
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),  # 转为tensor
                                              torchvision.transforms.Normalize(  # 进行数据的正则化
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size, shuffle=True)
    # 下载Fashion_MNIST训练集数据，并构建训练集数据载入器train_loader
    test_loader = torch.utils.data.DataLoader(  # 加载测试数据集
        torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size, shuffle=False)

    # 数据预览
    images, labels = next(iter(train_loader))
    plot_image(images, labels, 'sample')  # 只有CPU中的数据才能显示，默认就是

    # 定义用于处理数据的显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载和保存模型
    test_load = True  # 是否加载原有模型，如果网络有变化一定不能加载原来的
    if test_load:
        model = Net().to(device)  # 加载网络并输入到显卡
    else:
        if os.path.exists('model_Net.pkl'):  # 本文件夹下是否存在指定文件
            model = torch.load('model_Net.pkl').to(device)  # 加载保存好的网络，并进入显卡
            print('loaded')
        else:
            model = Net().to(device)
    # 定义损失函数loss function 和优化方式（采用SGD）
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(  # 每次遇到milestones中的epoch，做一次更新
        optimizer,  # optimizer （Optimizer）：要更改学习率的优化器；
        milestones=[12, 24],  # milestones（list）：递增的list，存放要更新lr的epoch；
        gamma=0.1)  # 学习率按区间更新 gamma（float）：更新lr的乘法因子

    # 开始训练
    for epoch in range(1, epoch + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        test(model, device, test_loader)


# 当.py文件直接运行时，该语句及以下的代码被执行，当.py被调用时，该语句及以下的代码不被执行
if __name__ == '__main__':
    main()
