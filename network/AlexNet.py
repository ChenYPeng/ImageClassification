import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 第一层卷积  [3, 224, 224] --> [64, 55, 55] --> [64, 27, 27]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # 第二层卷积  [64, 27, 27] --> [192, 27, 27] --> [192, 13, 13]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # 第三层卷积  [192, 13, 13] --> [384, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 第四层卷积  [384, 13, 13] --> [256, 13, 13]
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第五层卷积  [256, 13, 13] --> [ 256, 13, 13] --> [256, 6, 6]
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            # 第一层全连接层
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # 第二层全连接层
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 第三层全连接层（输出层）
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False):
    """Constructs a alexnet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(num_classes=1000)

    if pretrained:
        model.load_state_dict(load_url(model_urls['alexnet']))
    return model


if __name__ == '__main__':
    from torchsummary import summary

    alexnet = AlexNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = alexnet.to(device)
    summary(inputs, (3, 224, 224), batch_size=1, device="cuda")

    # input = torch.randn(2, 3, 224, 224)
    # model = alexnet(pretrained=True)
    # output = model(input)
    # print(output.shape)
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 第一层卷积  [3, 224, 224] --> [64, 55, 55] --> [64, 27, 27]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # 第二层卷积  [64, 27, 27] --> [192, 27, 27] --> [192, 13, 13]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # 第三层卷积  [192, 13, 13] --> [384, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 第四层卷积  [384, 13, 13] --> [256, 13, 13]
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第五层卷积  [256, 13, 13] --> [ 256, 13, 13] --> [256, 6, 6]
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            # 第一层全连接层
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # 第二层全连接层
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 第三层全连接层（输出层）
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False):
    """Constructs a alexnet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(num_classes=1000)

    if pretrained:
        model.load_state_dict(load_url(model_urls['alexnet']))
    return model


if __name__ == '__main__':
    from torchsummary import summary

    alexnet = AlexNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = alexnet.to(device)
    summary(inputs, (3, 224, 224), batch_size=1, device="cuda")

    # input = torch.randn(2, 3, 224, 224)
    # model = alexnet(pretrained=True)
    # output = model(input)
    # print(output.shape)
