import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn

'''
v1
'''
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Args:
            in_channels: 整个Inception的输入维度
            ch1x1:       分支1(1x1卷积核)的out_channels
            ch3x3red:    分支2(3x3卷积核)的in_channels
            ch3x3:       分支2(3x3卷积核)的out_channels
            ch5x5red:    分支3(5x5卷积核)的in_channels
            ch5x5:       分支3(5x5卷积核)的out_channels
            pool_proj:   分支4(1x1卷积核)的out_channels
        """
        super(Inception, self).__init__()
        # 分支1 -> 1x1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 分支2 -> 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )
        # 分支3 -> 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )
        # 分支4 -> 3x3 -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 保证输出大小等于输入大小
            BasicConv2d(in_channels, pool_proj, kernel_size=1)  # 1x1的卷积核，输入为in_channels，输出为pool_proj
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        # 拼接向量(从维度1开始拼接，维度0是batch)
        return torch.cat(outputs, dim=1)





