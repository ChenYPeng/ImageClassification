import torch  # 一般类和变量用小写；函数大写字母开头
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve
import os


batch_size = 64
train_loader = DataLoader(
    datasets.MNIST('mnist_data', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(
                           (0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST("mnist_data", train=False, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(
                           (0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)
test = True
if not test:
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, x.min(), x.max())
    plot_image(x, y, 'image sample')

##############################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################


class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(32 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        # tmp = torch.randn(2, 1, 28, 28)
        # out = self.conv_unit(tmp)
        # print('conv out:', out.shape)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, 32 * 8 * 8)
        logits = self.fc_unit(x)
        # logits = F.softmax(self.fc_unit(x)) #精度更低
        return logits


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk(64, 128, stride=1)
        self.blk2 = ResBlk(128, 256, stride=1)
        self.blk3 = ResBlk(256, 512, stride=1)
        self.blk4 = ResBlk(512, 512, stride=1)
        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


if test:
    net = FcNet().to(device)
else:
    if os.path.exists('model_Fc.pkl'):
        net = torch.load('model_Fc.pkl').to(device)
        print('loaded')
    else:
        ##########################
        net = FcNet().to(device)
        #########################
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_loss = []
for epoch in range(1):
    net.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        ################################
        x, y = x.to(device), y.to(device)
        ################################
        out = net(x)
        loss = criteon(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 100 == 0 and batch_idx != 0:
            print(epoch, batch_idx, loss.item())
            torch.save(net, 'model_Fc.pkl')
            print('saved')
plot_curve(train_loss)

total_correct = 0
for x, y in test_loader:
    net.eval()
    ################################
    x, y = x.to(device), y.to(device)
    ################################
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
#################################
x, y = x.to(device), y.to(device)
##################################
out = net(x)
pred = out.argmax(dim=1)
plot_image(x.cpu(), pred.cpu(), 'test')
