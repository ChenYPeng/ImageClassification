import torch        # 导入PyTorch深度学习库
import torchvision  # 导入数据集、网络模型管理库
import os           # 导入操作系统功能
from matplotlib import pyplot as plt    # 导入Python的2D绘图库


def main():
    batch_size = 32     # 并行处理的一批图像的量
    train_loader = torch.utils.data.DataLoader(     # 加载训练数据集
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   # 指定数据目录，类型为训练集，没有就下载
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),      # 转为tensor
                                       torchvision.transforms.Normalize(       # 进行数据的正则化
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(  # 加载测试数据集
        torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)
    # x, y = next(iter(train_loader))   # 取一批数据来显示其中的六个
    # print(x.shape, y.shape, x.min(), x.max())  # [64,1,28,28]
    # plot_image(x, y, 'image sample')  # 只有CPU中的数据才能显示，默认就是
    device = torch.device('cuda:0')     # 定义用于处理数据的显卡
    test = True                         # 是否加载原有模型，如果网络有变化一定不能加载原来的
    if test:
        net = LeNet5().to(device)       # 加载网络并输入到显卡
    else:
        if os.path.exists('model_Le.pkl'):                  # 本文件夹下是否存在指定文件
            net = torch.load('model_Le.pkl').to(device)     # 加载保存好的网络，并进入显卡
            print('loaded')
        else:
            net = LeNet5().to(device)
    criteon = torch.nn.CrossEntropyLoss()   # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)    # 定义优化器
    train_loss = []             # 保存损失变化过程的数组
    for epoch in range(1):      # 开始将所有训练数据，训练指定次数
        for batch_idx, (x, y) in enumerate(train_loader):       # 批次，输入，标签
            x, y = x.to(device), y.to(device)                   # 输入和标签进入显卡
            out = net(x)                # 用网络计算当前输出
            loss = criteon(out, y)      # 计算损失
            optimizer.zero_grad()       # 梯度清零
            loss.backward()             # 损失函数逆向求梯度
            optimizer.step()            # 更新一次网络
            train_loss.append(loss.item())                  # 增加一次损失数据
            if batch_idx % 100 == 0 and batch_idx != 0:     # 定期保存网络
                print(epoch, batch_idx, loss.item())
                torch.save(net, 'model_Le.pkl')
                print('saved')
    plot_curve(train_loss)      # 输出损失函数变化曲线
    total_correct = 0           # 正确项计数器，准备测试正确率
    for batch_idx, (x, y) in enumerate(test_loader):        # 开始测试测试集数据，批次，输入，标签
        x, y = x.to(device), y.to(device)       # 输入，标签进入显卡
        out = net(x)                            # 输入数据，用训练好的网络，计算输出
        pred = out.argmax(dim=1)                # 从输出中选出本次计算，概率最大的项作为预测结果
        correct = pred.eq(y).sum().float().item()       # 统计本批数据正确的次数
        total_correct += correct                        # 累加正确的次数
    total_num = len(test_loader.dataset)                # 总测试数据（不出错就是1万）
    print('total_num', total_num)
    acc = total_correct / total_num   # 计算正确率
    print('test acc:', acc)
    x, y = next(iter(test_loader))    # 取出一批测试数据
    x, y = x.to(device), y.to(device)
    out = net(x)                       # 计算这批测试数据的输出
    pred = out.argmax(dim=1)       # 取该批测试数据中每个输出中，概率最大的作为该个数据的预测值
    plot_image(x.cpu(), pred.cpu(), 'test')    # 从显卡回到CPU并显示其中6个


class FcNet(torch.nn.Module):   # 全连接网络（用类实现，继承自torch.nn.Module）
    def __init__(self):         # 定义构造函数
        super(FcNet, self).__init__()      # 构造父类构造函数
        self.fc_unit = torch.nn.Sequential(    # 定义一系列连续层
            torch.nn.Linear(28*28, 256),   # 线性层，28*28输入，256输出，输入必须和上一层输出同
            torch.nn.ReLU(),               # 非线性激活层
            torch.nn.Linear(256, 64),      # 线性层，256入，64出，入和上层输出同
            torch.nn.ReLU(),               # 非线性激活层
            torch.nn.Linear(64, 10),       # 线性层，64入，10出（数字10分类），入和上层输出同
        )

    def forward(self, x):                  # 定义前向网络
        x = x.view(x.size(0), 28 * 28)     # 将二维图像数据展开为一位数据，以供全连接网络用
        logits = self.fc_unit(x)           # 通过线性系列层，计算本批输出
        return logits


class LeNet5(torch.nn.Module):                  # 定义LeNet5网络类，继承自torch.nn.Module
    def __init__(self):                         # 定义构造函数
        super(LeNet5, self).__init__()          # 构造父类构造函数
        self.conv_unit = torch.nn.Sequential(   # 定义卷积系列连续层
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # 卷积层，1输入通道，16输出通道，卷积核3*3，步进1，填充边缘（图像大小不变）
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # 最大池化层，池化核2*2，步进2，填充边缘（图像减小一半）
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.fc_unit = torch.nn.Sequential(   # 定义线性系列连续层
            torch.nn.Linear(32*8*8, 32),      # 卷积系列层输出32通道的8*8展平后作为输入，输出32
            torch.nn.ReLU(),                  # 非线性激活层
            torch.nn.Linear(32, 10)           # 线性层，32入，10出（数字10分类），入和上层输出同
        )

    def forward(self, x):               # 定义前向网络
        x = self.conv_unit(x)           # 输入数据经过卷积系列层
        x = x.view(x.size(0), 32*8*8)   # 将上一层数据展平
        logits = self.fc_unit(x)        # 通过线性系列层，计算本批输出
        # logits = torch.nn.functional.softmax(self.fc_unit(x)) # 精度更低
        return logits


class ResBlk(torch.nn.Module):    # 定义残差块类，继承自torch.nn.Module
    def __init__(self, ch_in, ch_out, stride=1):    # 定义构造函数，需要输入输出层通道作为参数
        super(ResBlk, self).__init__()              # 构造父类构造函数
        self.conv1 = torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(ch_out)
        self.conv2 = torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(ch_out)
        self.extra = torch.nn.Sequential()         # 建一个空的额外层
        if ch_out != ch_in:       # 如果输出通道数不等于输入通道数，用额外层来修正
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(ch_out)        # 正则化数据
            )

    def forward(self, x):                           # 定义前向网络
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        out = self.extra(x) + out                   # 分支网络叠加
        out = torch.nn.functional.relu(out)
        return out


class ResNet18(torch.nn.Module):    # 定义残差网络类，继承自torch.nn.Module
    def __init__(self):                           # 定义构造函数
        super(ResNet18, self).__init__()          # 构造父类构造函数
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk(64, 128, stride=1)
        self.blk2 = ResBlk(128, 256, stride=1)
        self.blk3 = ResBlk(256, 512, stride=1)
        self.blk4 = ResBlk(512, 512, stride=1)
        self.outlayer = torch.nn.Linear(512*1*1, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x =torch.nn.functional.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def plot_curve(data):
    fig = plt.figure()       # 定义图形对象
    plt.plot(range(len(data)), data, color='blue')       # 关联x和y轴数据
    plt.legend(['value'], loc='upper right')             # 设置坐标轴方向
    plt.xlabel('step')                                   # 设置x，y标识符
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)          # 设置2行，3列子绘图区
        plt.tight_layout()                # 设置为固定布局
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':     # 下面函数只能在本文件执行，不能被外部文件调用
    main()


