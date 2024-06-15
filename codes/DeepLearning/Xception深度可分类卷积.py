import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import  nn

'''
分组卷积：对通道分组
卷积核的个数决定输出图像的通道数（深度）
三维图像卷积时，每个卷积核有3个通道，一个卷积核对整个三维图像进行卷积，最后得到一张与三维图像大小相同的特征图 当有两个卷积核时，则得到通道数为2的特征图
分组卷积：假设三维图像6个通道，每两个通道进行分组，每组要一个卷积核，则一共要三个卷积核，每个卷积核有两个通道，三个组每组产生一张特征图，最后得到三张特征图，即通道数为3
每个组的卷积核通道数为输入通道数//组数，每个组的卷积核的个数为输出通道数//组数
假设要从6通道转为12通道，分三组卷积，每组2通道，即每个组的卷积核的通道数为2，输出要12通道，3个组，每组的卷积核的个数为4个
'''
# 假设输入图像数据为一个 batch，大小为 (batch_size, channels, height, width)
batch_size = 1
channels = 6  # 假设输入图像有6个通道
height = 32
width = 32
input_data = torch.randn(batch_size, channels, height, width)  # 随机生成输入图像数据
# 定义分组卷积层
num_groups = 3  # 将输入通道分成3组
group_conv = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1, groups=num_groups)
# 执行分组卷积操作
output = group_conv(input_data)
# 查看输出的形状
print("分组卷积后的输出形状:", output.shape)
'''
input_data 是一个大小为 (1, 6, 32, 32) 的随机输入图像数据，表示一个样本，6个通道，每个通道大小为32x32像素。
group_conv 是一个分组卷积层，其参数设置为 groups=num_groups，表示将输入通道分成 num_groups 组，每组进行独立的卷积计算。
kernel_size=3, stride=1, padding=1 是卷积层的标准参数设置，保持输出特征图大小与输入特征图大小相同。
'''



'''
逐点卷积：每个卷积核尺寸为1*1*depth，卷积核个数为输出通道数
'''
# 假设输入数据为一个 batch，大小为 (batch_size, channels, height, width)
batch_size = 1
channels = 6
height = 32
width = 32
input_data = torch.randn(batch_size, channels, height, width)  # 随机生成输入数据
# 定义逐点卷积层
output_channels = 12  # 输出通道数
pointwise_conv = nn.Conv2d(channels, output_channels, kernel_size=1)
# 打印逐点卷积层的权重信息
print("逐点卷积层权重形状:", pointwise_conv.weight.shape)  # 形状为 (12, 6, 1, 1)，说明每个输出通道的卷积核大小为 1x1
# 执行逐点卷积操作
output = pointwise_conv(input_data)
# 查看输出的形状
print("逐点卷积后的输出形状:", output.shape)  # 输出形状为 (1, 12, 32, 32)，与输入的空间维度相同，通道数变为12



'''
深度卷积：分组卷积的groups=in_channels 每一个通道作为一组进行卷积 假设每组的卷积核个数为k，即每组的输出通道数为k，总输出通道数=k*in_channels
'''
# 假设输入数据为一个 batch，大小为 (batch_size, input_channels, height, width)
batch_size = 1
input_channels = 3  # 输入通道数
output_channels = 6  # 输出通道数
height = 32
width = 32
input_data = torch.randn(batch_size, input_channels, height, width)  # 随机生成输入数据
# 定义深度卷积层
conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
# 打印深度卷积层的权重信息
print("深度卷积层权重形状:", conv_layer.weight.shape)  # 形状为 (6, 3, 3, 3)，有 6 个卷积核，每个卷积核的大小是 3x3，输入通道数为 3
# 执行深度卷积操作
output = conv_layer(input_data)
# 查看输出的形状
print("深度卷积后的输出形状:", output.shape)  # 输出形状为 (1, 6, 32, 32)，输入通道数为 3，输出通道数为 6


'''
深度可分离卷积
'''
'''
深度卷积（Depthwise Convolution）：

这一步是对每个输入通道分别应用一个单独的卷积核。
输入是 inp 通道，输出也是 inp 通道，但每个通道都有自己的卷积核。
在PyTorch中，可以使用 Conv2d 的 groups 参数来实现分组卷积，其中 groups=inp 表示每个输入通道都有自己的卷积核。

逐点卷积（Pointwise Convolution）：

这一步是在深度卷积后应用的1x1卷积。
1x1卷积的目的是将深度卷积输出的通道数从 inp 调整为 oup。
这个卷积的核大小是 (1, 1)，通常使用 Conv2d 来实现。
'''

class SeparableConv2d(nn.Module):
    def __init__(self, inp, oup):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp)
        '''
        inp: 输入的通道数，即输入图像或特征图的深度。在这个例子中，输入通道数为 inp。
        inp: 输出的通道数，即卷积核的数量，也是输出特征图的深度。在这个例子中，输出通道数也为 inp，意味着每个输入通道都有一个对应的卷积核。
        kernel_size=3: 卷积核的大小，指定为一个3x3的矩阵。这表示卷积核在空间上是3x3的区域。
        stride=1: 卷积操作时的步长大小，默认为1。这意味着卷积核每次在输入图像上水平和垂直方向上移动1个像素进行卷积操作。
        padding=1: 输入图像的每一条边缘都填充了1个像素宽度的零像素。这样做是为了确保卷积操作后输出特征图的大小与输入特征图的大小相同。填充有助于在卷积过程中保留边缘信息。
        groups=inp: 这个参数决定了输入和输出通道之间的连接方式。在这个例子中，每个输入通道都被分配一个单独的卷积核进行处理，因此 groups=inp 意味着每个输入通道有自己的卷积核，这是深度可分离卷积的一部分。
        '''
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu = nn.ReLU(inplace=True)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(oup)

    def forward(self, x):
        print(x.shape)
        x = self.depthwise(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        print(x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# 使用示例：
inp_channels = 64
out_channels = 128
sep_conv = SeparableConv2d(inp_channels, out_channels)
print(sep_conv)



'''
depthwise_conv 是一个深度可分离卷积的深度卷积层，其参数设置为 kernel_size=3, stride=1, padding=1, groups=channels。
padding=1 表示在输入的每一边都填充1个像素的零，保持了输出特征图的大小与输入特征图相同。
pointwise_conv 是逐点卷积层，其卷积核大小为 1x1，它不会改变特征图的空间维度。
'''
# 假设输入图像数据为一个 batch，大小为 (batch_size, channels, height, width)
batch_size = 1
channels = 3  # 假设输入图像有3个通道（RGB图像）
height = 64
width = 64
input_data = torch.randn(batch_size, channels, height, width)  # 随机生成输入图像数据

# 定义深度可分离卷积层
depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)

# 执行深度可分离卷积操作
depthwise_output = depthwise_conv(input_data)
pointwise_output = pointwise_conv(depthwise_output)

# 查看输出的形状
print("深度可分离卷积后的输出形状:", pointwise_output.shape)




# 假设输入图像数据为一个 batch，大小为 (batch_size, channels, height, width)
batch_size = 1
channels = 3  # 假设输入图像有3个通道（RGB图像）
height = 64
width = 64
input_data = torch.randn(batch_size, channels, height, width)  # 随机生成输入图像数据

# 定义深度可分离卷积层
depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
pointwise_conv = nn.Conv2d(channels, 6, kernel_size=1)  # 输出通道数为6

# 执行深度可分离卷积操作
depthwise_output = depthwise_conv(input_data)
pointwise_output = pointwise_conv(depthwise_output)

# 查看输出的形状
print("深度可分离卷积后的输出形状:", pointwise_output.shape)




# class Xception(nn.Module):
#     def __init__(self,inp,oup):
#         super(Xception, self).__init__()
#         # depthwise
#         self.conv1 = Conv2d(inp, inp, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=inp)
#         self.bn1 = BatchNorm2d(inp)  # 输入为上一层输出的通道数
#         # pointwise
#         self.conv2 = Conv2d(inp, oup, (1, 1))  # Stride of the convolution. Default: 1
#         self.bn2 = BatchNorm2d(oup)
#         self.relu = nn.ReLU()
#
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.bn1(output)
#         output = self.relu(output)
#         output = self.conv2(output)
#         output = self.bn2(output)
#         output = self.relu(output)
#         return output




import torch.nn as nn

class SeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        ## 3*3卷积
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels, ##一个3*3卷积核只处理一个通道，所以输入是多少就有多少个3*3卷积核
            bias=False,
            **kwargs
        )
        ## 1*1卷积
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    ## 先经过3*3卷积，再经过1*1卷积？？？？？论文中不是先经过1*1再3*3吗
    ## 解释：差距其实不大，因为网络是层级结构
    ## 每一层是堆叠上去的 1*1 3*3 1*1 3*3 1*1 3*3 他可能就是先框中了第2个和第3个，即先3*3再1*1了？？？？
    ## 可分离卷积：1*1主要用于处理channel通道上的信息,3*3主要用于处理空间上的信息

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EntryFlow(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        ## 残差连接，就是论文图片中的那个分支
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        #no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut ##两个结果  拼接
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        shortcut = self.shortcut(x)
        return shortcut + residual


class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.middel_block = self._make_flow(block, 8) ##重复8次

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)

        return output


class Xception(nn.Module):

    def __init__(self, block, num_classes=100):
        super().__init__()
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1) ##把[[........]] 调整成 [..........]
        x = self.fc(x) ##最后还有一个全连接层
        return x


'''
torch.nn.Conv2d(
    in_channels=in_channels,  # 输入的通道数，例如RGB图像输入为3
    out_channels=out_channels,  # 输出的通道数，即卷积核的数量
    kernel_size=kernel_size,  # 卷积核的尺寸，可以是单个整数或元组，如(3, 3)
    stride=stride,  # 卷积操作时每次移动的步长，默认为1
    padding=0,  # 输入的每一条边补充0的层数
    dilation=1,  # 卷积核元素之间的间距，默认为1
    groups=1,  # 输入和输出之间连接的组数，控制输入和输出之间的联系
    bias=True,  # 是否添加偏置项，默认为True
    padding_mode='zeros',  # padding的模式，可以是'zeros'或'circular'
)
'''