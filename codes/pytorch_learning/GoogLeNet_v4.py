import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn

'''
v4
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
    """
    stem block for Inception-v4
    """

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


class InceptionV4_A(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, out_channels_3red, out_channels_3,
                 out_channels_4red, out_channels_4):
        """
        Args:
            in_channels:        整个Inception的输入维度
            out_channels_1:     分支1(1x1卷积核)的out_channels
            out_channels_2:     分支2(1x1卷积核)的out_channels
            out_channels_3red:  分支3(1x1卷积核)的out_channels
            out_channels_3:     分支3(3x3卷积核)的out_channels
            out_channels_4red:  分支4(1x1卷积核)的out_channels
            out_channels_4:     分支4(3x3卷积核)的out_channels
        """
        super(InceptionV4_A, self).__init__()

        # 分支1：avg -> 1x1
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_1, kernel_size=1)
        )

        # 分支2：1x1
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2, kernel_size=1)
        )

        # 分支3：(1x1 -> 3x3)
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_3red, kernel_size=1),
            ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=3, stride=1, padding=1)
        )

        # 分支4：(1x1 -> 3x3 -> 3x3)
        self.branch4 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_4red, kernel_size=1),
            ConvBNReLU(out_channels_4red, out_channels_4, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(out_channels_4, out_channels_4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV4_B(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2,
                 out_channels_3_1x1, out_channels_3_1x7, out_channels_3,
                 out_channels_4_1x1, out_channels_4_1x7_1, out_channels_4_7x1_1,
                 out_channels_4_1x7_2, out_channels_4_7x1_2):
        super(InceptionV4_B, self).__init__()

        # 分支1：(AvgPool->1x1)
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_1, kernel_size=1)
        )

        # 分支2：(1x1)
        self.branch2 = ConvBNReLU(in_channels, out_channels_2, kernel_size=1)

        # 分支3：(1x1->1x7->7x1)
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_3_1x1, kernel_size=1),
            ConvBNReLU(out_channels_3_1x1, out_channels_3_1x7, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(out_channels_3_1x7, out_channels_3, kernel_size=[7, 1], padding=[3, 0])
        )

        # 分支4：(1x1->1x7->7x1->1x7->7x1)
        self.branch4 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_4_1x1, kernel_size=1),
            ConvBNReLU(out_channels_4_1x1, out_channels_4_1x7_1, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(out_channels_4_1x7_1, out_channels_4_7x1_1, kernel_size=[7, 1], padding=[3, 0]),
            ConvBNReLU(out_channels_4_7x1_1, out_channels_4_1x7_2, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(out_channels_4_1x7_2, out_channels_4_7x1_2, kernel_size=[7, 1], padding=[3, 0])
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV4_C(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2,
                 out_channels_3red, out_channels_3,
                 out_channels_4_1x1, out_channels_4_1x3_1, out_channels_4_3x1_1,
                 out_channels_4_3x1_2, out_channels_4_1x3_2):
        super(InceptionV4_C, self).__init__()

        # 分支1：(AvgPool->1x1)
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_1, kernel_size=1)
        )

        # 分支2：(1x1)
        self.branch2 = ConvBNReLU(in_channels, out_channels_2, kernel_size=1)

        # 分支3：(1x1->两个分支：①1x3；②3x1)
        self.branch3_conv1x1 = ConvBNReLU(in_channels, out_channels_3red, kernel_size=1)
        self.branch3_conv1x3 = ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=[1, 3], padding=[0, 1])
        self.branch3_conv3x1 = ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=[3, 1], padding=[1, 0])

        # 分支4：(1x1->1x3->3x1->两个分支：①1x3；②3x1)
        self.branch4_step1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_4_1x1, kernel_size=1),
            ConvBNReLU(out_channels_4_1x1, out_channels_4_1x3_1, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels_4_1x3_1, out_channels_4_3x1_1, kernel_size=[3, 1], padding=[1, 0])
        )
        self.branch4_conv3x1 = ConvBNReLU(out_channels_4_3x1_1, out_channels_4_3x1_2, kernel_size=[3, 1],
                                          padding=[1, 0])
        self.branch4_conv1x3 = ConvBNReLU(out_channels_4_3x1_1, out_channels_4_1x3_2, kernel_size=[1, 3],
                                          padding=[0, 1])

    def forward(self, x):
        # 分支1
        branch1 = self.branch1(x)
        # 分支2
        branch2 = self.branch2(x)
        # 分支3
        branch3_tmp = self.branch3_conv1x1(x)
        branch3 = torch.cat([self.branch3_conv1x3(branch3_tmp), self.branch3_conv3x1(branch3_tmp)], dim=1)
        # 分支4
        branch4_tmp = self.branch4_step1(x)
        branch4 = torch.cat([self.branch4_conv3x1(branch4_tmp), self.branch4_conv1x3(branch4_tmp)], dim=1)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


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
    def __init__(self, in_channels, out_channels_2_1x1, out_channels_2_3x3,
                 out_channels_3_1x1, out_channels_3_1x7, out_channels_3_7x1, out_channels_3_3x3):
        super(Reduction_B, self).__init__()

        # 分支1：MaxPool
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 分支2：(3x3)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2_1x1, kernel_size=1),
            ConvBNReLU(out_channels_2_1x1, out_channels_2_3x3, kernel_size=3, stride=2)
        )

        # 分支3：(1x1->1x7->7x1)
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_3_1x1, kernel_size=1),
            ConvBNReLU(out_channels_3_1x1, out_channels_3_1x7, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(out_channels_3_1x7, out_channels_3_7x1, kernel_size=[7, 1], padding=[3, 0]),
            ConvBNReLU(out_channels_3_7x1, out_channels_3_3x3, kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)


class InceptionV4_C(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2,
                 out_channels_3red, out_channels_3,
                 out_channels_4_1x1, out_channels_4_1x3_1, out_channels_4_3x1_1,
                 out_channels_4_3x1_2, out_channels_4_1x3_2):
        super(InceptionV4_C, self).__init__()

        # 分支1：(AvgPool->1x1)
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_1, kernel_size=1)
        )

        # 分支2：(1x1)
        self.branch2 = ConvBNReLU(in_channels, out_channels_2, kernel_size=1)

        # 分支3：(1x1->两个分支：①1x3；②3x1)
        self.branch3_conv1x1 = ConvBNReLU(in_channels, out_channels_3red, kernel_size=1)
        self.branch3_conv1x3 = ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=[1, 3], padding=[0, 1])
        self.branch3_conv3x1 = ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=[3, 1], padding=[1, 0])

        # 分支4：(1x1->1x3->3x1->两个分支：①1x3；②3x1)
        self.branch4_step1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_4_1x1, kernel_size=1),
            ConvBNReLU(out_channels_4_1x1, out_channels_4_1x3_1, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels_4_1x3_1, out_channels_4_3x1_1, kernel_size=[3, 1], padding=[1, 0])
        )
        self.branch4_conv3x1 = ConvBNReLU(out_channels_4_3x1_1, out_channels_4_3x1_2, kernel_size=[3, 1],
                                          padding=[1, 0])
        self.branch4_conv1x3 = ConvBNReLU(out_channels_4_3x1_1, out_channels_4_1x3_2, kernel_size=[1, 3],
                                          padding=[0, 1])

    def forward(self, x):
        # 分支1
        branch1 = self.branch1(x)
        # 分支2
        branch2 = self.branch2(x)
        # 分支3
        branch3_tmp = self.branch3_conv1x1(x)
        branch3 = torch.cat([self.branch3_conv1x3(branch3_tmp), self.branch3_conv3x1(branch3_tmp)], dim=1)
        # 分支4
        branch4_tmp = self.branch4_step1(x)
        branch4 = torch.cat([self.branch4_conv3x1(branch4_tmp), self.branch4_conv1x3(branch4_tmp)], dim=1)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

