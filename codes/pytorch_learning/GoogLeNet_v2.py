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



class InceptionV2_A(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2,
                 out_channels_3, out_channels_4):
        """
        Args:
            in_channels:        整个Inception的输入维度
            out_channels_1red:  分支1(1x1卷积核)的out_channels
            out_channels_1:     分支1(3x3卷积核)的out_channels
            out_channels_2red:  分支2(1x1卷积核)的out_channels
            out_channels_2:     分支2(3x3卷积核)的out_channels
            out_channels_3:     分支3(1x1卷积核)的out_channels
            out_channels_4:     分支4(1x1卷积核)的out_channels
        """
        super(InceptionV2_A, self).__init__()

        # 分支1：(1x1->3x3->3x3)
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_1red, kernel_size=1),
            ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=3, padding=1),  # 保证输出大小等于输入大小
            ConvBNReLU(out_channels_1, out_channels_1, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        # 分支2：(1x1->3x3)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        # 分支3：(MaxPool->1x1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_3, kernel_size=1)  # 保证输出大小等于输入大小
        )

        # 分支4（1x1）
        self.branch4 = ConvBNReLU(in_channels, out_channels_4, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



class InceptionV2_B(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2,
                 out_channels_3, out_channels_4):
        """
        Args:
            in_channels:        整个Inception的输入维度
            out_channels_1red:  分支1(1x1卷积核)的out_channels
            out_channels_1:     分支1(3x1卷积核)的out_channels
            out_channels_2red:  分支2(1x1卷积核)的out_channels
            out_channels_2:     分支2(3x1卷积核)的out_channels
            out_channels_3:     分支3(1x1卷积核)的out_channels
            out_channels_4:     分支4(1x1卷积核)的out_channels
        """
        super(InceptionV2_B, self).__init__()

        # 分支1：(1x1->1x3->3x1->1x3->3x1)
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_1red, kernel_size=1),
            # 使用1x3卷积核时，需要分别设置padding，以保证WxH不发生改变
            ConvBNReLU(out_channels_1red, out_channels_1red, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels_1red, out_channels_1red, kernel_size=[3, 1], padding=[1, 0]),
            ConvBNReLU(out_channels_1red, out_channels_1red, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=[3, 1], padding=[1, 0])
        )

        # 分支2：(1x1->1x3->3x1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2red, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[3, 1], padding=[1, 0])
        )

        # 分支3：(MaxPool->1x1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_3, kernel_size=1)
        )

        # 分支4：(1x1)
        self.branch4 = ConvBNReLU(in_channels, out_channels_4, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



class InceptionV2_C(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2,
                 out_channels_3, out_channels_4):
        """
        Args:
            in_channels:        整个Inception的输入维度
            out_channels_1red:  分支1(1x1卷积核)的out_channels
            out_channels_1:     分支1(1x3与3x1卷积核)的out_channels
            out_channels_2red:  分支2(1x1卷积核)的out_channels
            out_channels_2:     分支2(1x3与3x1卷积核)的out_channels
            out_channels_3:     分支3(1x1卷积核)的out_channels
            out_channels_4:     分支4(1x1卷积核)的out_channels
        """
        super(InceptionV2_C, self).__init__()

        # 分支1：(1x1->3x3->两个分支：①1x3；②3x1)
        self.branch1_conv1x1 = ConvBNReLU(in_channels, out_channels_1red, kernel_size=1)
        self.branch1_conv3x3 = ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=3, padding=1)
        self.branch1_conv1x3 = ConvBNReLU(out_channels_1, out_channels_1, kernel_size=[1, 3], padding=[0, 1])
        self.branch1_conv3x1 = ConvBNReLU(out_channels_1, out_channels_1, kernel_size=[3, 1], padding=[1, 0])

        # 分支2：(1x1->两个分支：①1x3；②3x1)
        self.branch2_conv1x1 = ConvBNReLU(in_channels, out_channels_2red, kernel_size=1)
        self.branch2_conv1x3 = ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[1, 3], padding=[0, 1])
        self.branch2_conv3x1 = ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[3, 1], padding=[1, 0])

        # 分支3：(MaxPool->1x1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_3, kernel_size=1)
        )

        # 分支4：(1x1)
        self.branch4 = ConvBNReLU(in_channels, out_channels_4, kernel_size=1)

    def forward(self, x):
        # 分支1
        branch1_tmp = self.branch1_conv1x1(x)
        branch1_tmp = self.branch1_conv3x3(branch1_tmp)
        branch1 = torch.cat([self.branch1_conv1x3(branch1_tmp), self.branch1_conv3x1(branch1_tmp)], dim=1)

        # 分支2
        branch2_tmp = self.branch2_conv1x1(x)
        branch2 = torch.cat([self.branch2_conv1x3(branch2_tmp), self.branch2_conv3x1(branch2_tmp)], dim=1)

        # 分支3
        branch3 = self.branch3(x)

        # 分支4
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



class InceptionV2_D(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2):
        super(InceptionV2_D, self).__init__()

        # 分支1：(1x1->3x3->3x3)
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_1red, kernel_size=1),
            ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(out_channels_1, out_channels_1, kernel_size=3, stride=2, padding=1)
        )

        # 分支2：(1x1->3x3)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=3, stride=2, padding=1)
        )

        # 分支3：(1x1)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)





