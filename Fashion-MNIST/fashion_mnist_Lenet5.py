import argparse  # 加载处理命令行参数的库
import torch  # 引入相关的包
import torch.nn as nn  # 指定torch.nn别名nn
import torch.nn.functional as F  # 引用神经网络常用函数包，不具有可学习的参数
from torch import optim
from torchvision import datasets, transforms  # 加载pytorch官方提供的dataset
from matplotlib import pyplot as plt    # 导入Python的2D绘图库
from torch.utils.data import DataLoader


# 网络模型
class FcNet(nn.Module):   # 全连接网络（用类实现，继承自torch.nn.Module）
    def __init__(self):         # 定义构造函数
        super(FcNet, self).__init__()      # 构造父类构造函数
        self.fc_unit = nn.Sequential(    # 定义一系列连续层
            nn.Linear(28 * 28, 256),   # 线性层，28*28输入，256输出，输入必须和上一层输出同
            nn.ReLU(),               # 非线性激活层
            nn.Linear(256, 64),      # 线性层，256入，64出，入和上层输出同
            nn.ReLU(),               # 非线性激活层
            nn.Linear(64, 10),       # 线性层，64入，10出（数字10分类），入和上层输出同
        )

    def forward(self, x):                  # 定义前向网络
        x = x.view(x.size(0), 28 * 28)     # 将二维图像数据展开为一位数据，以供全连接网络用
        logits = self.fc_unit(x)           # 通过线性系列层，计算本批输出
        return logits


class LeNet5(nn.Module):                  # 定义LeNet5网络类，继承自torch.nn.Module
    def __init__(self):                         # 定义构造函数
        super(LeNet5, self).__init__()          # 构造父类构造函数
        self.conv_unit = nn.Sequential(   # 定义卷积系列连续层
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # 卷积层，1输入通道，16输出通道，卷积核3*3，步进1，填充边缘（图像大小不变）
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # 最大池化层，池化核2*2，步进2，填充边缘（图像减小一半）
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.fc_unit = nn.Sequential(   # 定义线性系列连续层
            nn.Linear(32 * 8 * 8, 32),      # 卷积系列层输出32通道的8*8展平后作为输入，输出32
            nn.ReLU(),                  # 非线性激活层
            nn.Linear(32, 10)           # 线性层，32入，10出（数字10分类），入和上层输出同
        )

    def forward(self, x):               # 定义前向网络
        x = self.conv_unit(x)           # 输入数据经过卷积系列层
        x = x.view(x.size(0), 32 * 8 * 8)   # 将上一层数据展平
        logits = self.fc_unit(x)        # 通过线性系列层，计算本批输出
        # logits = torch.nn.functional.softmax(self.fc_unit(x)) # 精度更低
        return logits


class ResBlk(torch.nn.Module):    # 定义残差块类，继承自torch.nn.Module
    def __init__(self, ch_in, ch_out, stride=1):    # 定义构造函数，需要输入输出层通道作为参数
        super(ResBlk, self).__init__()              # 构造父类构造函数
        self.conv1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()         # 建一个空的额外层
        if ch_out != ch_in:       # 如果输出通道数不等于输入通道数，用额外层来修正
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)        # 正则化数据
            )

    def forward(self, x):                           # 定义前向网络
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        out = self.extra(x) + out                   # 分支网络叠加
        out = F.relu(out)
        return out


class ResNet18(torch.nn.Module):    # 定义残差网络类，继承自torch.nn.Module
    def __init__(self):                           # 定义构造函数
        super(ResNet18, self).__init__()          # 构造父类构造函数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk(64, 128, stride=1)
        self.blk2 = ResBlk(128, 256, stride=1)
        self.blk3 = ResBlk(256, 512, stride=1)
        self.blk4 = ResBlk(512, 512, stride=1)
        self.out_layer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.out_layer(x)
        return x


# 定义训练函数
def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    num_correct = 0
    train_loss_list, train_acc_list = [], []  # 定义两个保存损失变化过程的数组
    for batch_idx, (images, labels) in enumerate(train_loader):  # 批次，输入，标签
        images, labels = images.to(device), labels.to(device)   # 输入和标签进入显卡
        optimizer.zero_grad()  # 清空梯度，在每次优化前都要进行此操作
        output = model(images)  # 用网络计算当前输出
        # negative log likelihood loss(nll_loss), sum up batch cross entropy
        loss = F.nll_loss(output, labels)  # 调用交叉熵损失函数
        loss.backward()   # 损失的反向传播
        optimizer.step()  # 根据parameter的梯度更新parameter的值
        scheduler.step()
        train_loss_list.append(train_loss / (len(train_loader)))  # 增加一次损失数据
        train_loss += float(loss.item())
        pred = output.argmax(dim=1)
        num_correct += torch.eq(pred, labels).sum().float().item()
        train_acc_list.append(100 * train_acc / (len(train_loader)))  # 增加一次损失数据
        train_acc = num_correct / len(train_loader.dataset)  # 计算实际损失
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        # batch_size = 样本总数/训练次数 = len(train_loader.dataset) / len(train_loader)
        # accuracy =


# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 无需计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            # sum up batch loss
            test_loss += F.nll_loss(output, labels, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    plot_image(images.cpu(), pred.cpu(), 'test')  # 从显卡回到CPU并显示其中6个
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 定义显示函数
def plot_curve(image):
    fig = plt.figure()       # 定义图形对象
    plt.plot(range(len(image)), image, color='blue')       # 关联x和y轴数据
    plt.legend(['value'], loc='upper right')             # 设置坐标轴方向
    plt.xlabel('step')                                   # 设置x，y标识符
    plt.ylabel('value')
    plt.show()


def plot_image(image, label, name):
    label_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)          # 设置3行，3列子绘图区
        plt.tight_layout()                # 设置为固定布局
        plt.imshow(image[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label_list[label[i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 定义主函数
# 创建 ArgumentParser() 对象
# 调用 add_argument() 方法添加参数
# 使用 parse_args() 解析添加的参数
def main():
    # Training settings
    # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser(description='PyTorch Fashion_MNIST Example')  # 创建一个对象
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # 增加一个叫batch-size的参数，类型必须是int
                        help='input batch size for training (default: 64)')  # type：参数的类型
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # default：参数默认值
                        help='input batch size for testing (default: 1000)')    # choices: 这个参数用来检查输入参数的范围。
    parser.add_argument('--epochs', type=int, default=10, metavar='N',  # nargs: 当选项后接受多个或者0个参数时需要这个来指定。
                        help='number of epochs to train (default: 10)')  #  help=”帮助信息”
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',   # metavar：这个参数用于help 信息输出中
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 定义超参数

    # 加载数据
    # 下载Fashion_MNIST训练集数据，并构建训练集数据载入器train_loader,
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True,
                              transform=transforms.Compose([  # 对数据进行预处理
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # 下载Fashion_MNIST训练集数据，并构建训练集数据载入器train_loader
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './fashionmnist_data/',  # root 的主目录 用于指定我们载入的数据集名称
            train=False,   # True = 训练集, False = 测试集
            transform=transforms.Compose([
                transforms.ToTensor(),  # 进行数据的正则化
                transforms.Normalize(
                    (0.5, ), (0.5, ))  # 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
            ])),
        batch_size=args.test_batch_size,  # 每个batch加载多少个样本(默认: 1)
        shuffle=True,  # 设置为True时会在每个epoch重新打乱数据(默认: False)
        **kwargs)

    # 开始训练
    # 数据可视化
    images, labels = next(iter(train_loader))  # 取一批数据来显示其中的六个
    print(images.shape, labels.shape, images.min(), images.max())  # [64,1,28,28]
    plot_image(images, labels, 'sample')  # 只有CPU中的数据才能显示，默认就是

    # 定义损失函数loss function 和优化方式（采用SGD）
    model = LeNet5().to(device)
    # optimizer存储了所有parameters的引用，每个parameter都包含gradient
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(  # 每次遇到milestones中的epoch，做一次更新
        optimizer,            # optimizer （Optimizer）：要更改学习率的优化器；
        milestones=[12, 24],  # milestones（list）：递增的list，存放要更新lr的epoch；
        gamma=0.1)            # 学习率按区间更新 gamma（float）：更新lr的乘法因子

    # 开始训练
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test(args, model, device, test_loader)
    # 保存网络
    if (args.save_model):
        torch.save(model.state_dict(), "model_LeNet5.pt")


# 当.py文件直接运行时，该语句及以下的代码被执行，当.py被调用时，该语句及以下的代码不被执行
if __name__ == '__main__':
    main()
