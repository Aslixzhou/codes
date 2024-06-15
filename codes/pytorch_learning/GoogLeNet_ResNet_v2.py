import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn

'''
v2
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x



class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()

        # 连续3个3x3的卷积核
        self.step1 = nn.Sequential(
            # 299x299x3 -> 149x149x32
            ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            # 149x149x32 -> 147x147x32
            ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            # 147x147x32 -> 147x147x64
            ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        # 分支1：147x147x64 -> 72x72x64
        self.step2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 分支2：147x147x64 -> 72x72x96
        self.step2_conv = ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=2)

        # 分支1：1x1+3x3
        self.step3_1 = nn.Sequential(
            ConvBNReLU(in_channels=160, out_channels=64, kernel_size=1, stride=1),
            ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        )
        # 分支2：1x1+7x1+1x7+3x3
        self.step3_2 = nn.Sequential(
            ConvBNReLU(in_channels=160, out_channels=64, kernel_size=1, stride=1),
            ConvBNReLU(in_channels=64, out_channels=64, kernel_size=[7, 1], padding=[3, 0]),
            ConvBNReLU(in_channels=64, out_channels=64, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        )

        # 分支1：池化
        self.step4_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 分支2：3x3
        self.step4_conv = ConvBNReLU(in_channels=192, out_channels=192, kernel_size=3, stride=2)

    def forward(self, x):
        out = self.step1(x)

        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)

        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)

        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)

        outputs = [tmp1, tmp2]
        return torch.cat(outputs, 1)


class Inception_A(nn.Module):

    def __init__(self, in_channels, b1, b2_1x1, b2_3x3, b3_1x1, b3_3x3_1, b3_3x3_2, n1_linear):
        super(Inception_A, self).__init__()

        # 分支1：
        self.branch1 = ConvBNReLU(in_channels, b1, kernel_size=1, stride=1)

        # 分支2：1x1 -> 3x3
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, b2_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b2_1x1, b2_3x3, kernel_size=3, stride=1, padding=1)
        )

        # 分支3：1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, b3_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b3_1x1, b3_3x3_1, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(b3_3x3_1, b3_3x3_2, kernel_size=3, stride=1, padding=1)
        )

        # 1x1Conv
        self.conv_linear = nn.Conv2d(b1 + b2_3x3 + b3_3x3_2, n1_linear, 1, 1, 0, bias=True)

        """
            因为这里需要将原始输入通过直连边连接到输出部分，所以需要判断in_channels和n1_linear的关系
        """
        # 如果in_channels==n1_linear，则不进行short_cut
        self.short_cut = nn.Sequential()
        # 如果in_channels!=n1_linear，则进行short_cut，把原始输入维度转为n1_linear
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)

        # 残差连接
        out += self.short_cut(x)

        out = self.relu(out)
        return out


class Inception_B(nn.Module):
    def __init__(self, in_channels, b1, b2_1x1, b2_1x7, b2_7x1, n1_linear):
        super(Inception_B, self).__init__()

        # 分支1：
        self.branch1 = ConvBNReLU(in_channels, b1, kernel_size=1, stride=1)

        # 分支2：
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, b2_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b2_1x1, b2_1x7, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(b2_1x7, b2_7x1, kernel_size=[7, 1], padding=[3, 0])
        )

        # 1x1Conv
        self.conv_linear = nn.Conv2d(b1 + b2_7x1, n1_linear, 1, 1, 0, bias=False)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)

        # 残差连接
        out += self.short_cut(x)

        out = self.relu(out)
        return out


class Inception_C(nn.Module):
    def __init__(self, in_channels, b1, b2_1x1, b2_1x3, b2_3x1, n1_linear):
        super(Inception_C, self).__init__()

        # 分支1：
        self.branch1 = ConvBNReLU(in_channels, b1, kernel_size=1, stride=1)

        # 分支2：
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, b2_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b2_1x1, b2_1x3, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(b2_1x3, b2_3x1, kernel_size=[3, 1], padding=[1, 0])
        )

        # 1x1Conv
        self.conv_linear = nn.Conv2d(b1 + b2_3x1, n1_linear, 1, 1, 0, bias=False)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)

        # 残差连接
        out += self.short_cut(x)

        out = self.relu(out)
        return out


class Reduction_A(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()

        # 分支1：MaxPool
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 分支2：(3x3)
        self.branch2 = ConvBNReLU(in_channels, n, kernel_size=3, stride=2)

        # 分支3：(1x1->3x3->3x3)
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, k, kernel_size=1),
            ConvBNReLU(k, l, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(l, m, kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)


class Reduction_B(nn.Module):
    def __init__(self, in_channels, b2_1x1, b2_3x3, b3_1x1, b3_3x3, b4_1x1, b4_3x3_1, b4_3x3_2):
        super(Reduction_B, self).__init__()

        # 分支1：
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 分支2：
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, b2_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b2_1x1, b2_3x3, kernel_size=3, stride=2)
        )

        # 分支3：
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, b3_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b3_1x1, b3_3x3, kernel_size=3, stride=2)
        )

        # 分支4：
        self.branch4 = nn.Sequential(
            ConvBNReLU(in_channels, b4_1x1, kernel_size=1, stride=1),
            ConvBNReLU(b4_1x1, b4_3x3_1, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(b4_3x3_1, b4_3x3_2, kernel_size=3, stride=2),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class GoogLeNet_ResNet_V2(nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(GoogLeNet_ResNet_V2, self).__init__()

        # 整体主干网络
        self.stem = Stem()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()

        # 输出部分：平均池化->全连接层
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2144, num_classes)

        if init_weights:
            self._initialize_weights()

    # 制造5层Inception-A
    def __make_inception_A(self):
        layers = []
        for _ in range(5):
            layers.append(Inception_A(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    # 制造1层Reduction-A
    def __make_reduction_A(self):
        return Reduction_A(384, 256, 256, 384, 384)

    # 制造10层Inception-B
    def __make_inception_B(self):
        layers = []
        for _ in range(10):
            layers.append(Inception_B(1152, 192, 128, 160, 192, 1152))
        return nn.Sequential(*layers)

    # 制造1层Reduction-B
    def __make_reduction_B(self):
        return Reduction_B(1152, 256, 384, 256, 288, 256, 288, 320)

    # 制造5层Inception-C
    def __make_inception_C(self):
        layers = []
        for _ in range(5):
            layers.append(Inception_C(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




