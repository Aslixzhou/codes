import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision
from datetime import datetime
from torchsummary import summary

'''
每个dense_block中存在num_layers个conv_block，
dense_block中的in_channel是输入dense_block的通道数，
dense_block中的out_channel是dense_block中的每个conv_block输出的通道数，
经过一个dense_block后，通道数从原来的in_channel变换到in_channel + num_layers * out_channel，
举例：一个dense_block的输入channel=3（in_channel=3），num_layers=2，out_channel=10，则经过一个dense_block后输出的channel = 3 + 10 * 2 = 23

'''

def conv_block(in_channel, out_channel): # RN+ReLU+Conv_3×3 输入通道 输出通道
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

# 稠密块 dense_block 由多个conv_block 组成，dense_block的每块使⽤用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class dense_block(nn.Module):
    # growth_rate即output_channel
    def __init__(self, in_channel, out_channel, num_layers): # out_channel 即学习率growth_rate 每个卷积块的输出通道数
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers): # 0 1 2 ... num_layers-1
            block.append(
                conv_block(in_channel=out_channel * i + in_channel, out_channel=out_channel)
            )
            '''
            追加conv_block的方式：
                假定dense_block的输入通道3 输出通道10 层数2
                i=0 conv_block0: channel=in_channel=3（输入） out_channel=out_channel=10（输出）
                i=1 conv_block1: channel=in_channel+1*10=13  out_channel=out_channel=10（输出）
            一个dense_block：将图像通道从3-->10
            '''
        self.net = nn.Sequential(*block)

    def forward(self, x):
        print(x.shape) # torch.Size([4, 3, 8, 8])
        for layer in self.net: # 对在dense_block的每一层
            out = layer(x) # out逐层输出通道数为：10 10
            # 追加拼接：x（3）+ 10 + 10 = 23
            x = torch.cat((out, x), dim=1) # 将每个卷积块的输出与输入在通道维度上拼接 返回拼接后的张量
            # print(x.shape) # torch.Size([4, 13, 8, 8]) torch.Size([4, 23, 8, 8])
        print(x.shape)
        return x # torch.Size([4, 23, 8, 8])

blk = dense_block(in_channel=3, out_channel=10, num_layers=2)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)  # torch.Size([4, 43, 8, 8])

# 每个稠密块会带来通道数的增加，在稠密块之间加上过渡块transition_block，控制模型复杂度
def transition_block(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=1), # 1×1卷积层减少通道数
        nn.AvgPool2d(kernel_size=2, stride=2) # 步幅为2的平均汇聚层减半高宽 进一步降低模型复杂度
    )
    return trans_layer

blk = transition_block(in_channel=23, out_channel=10) # 输入通道23 输出通道10
print(blk(Y).shape)  # torch.Size([4, 10, 4, 4])

class DenseNet(nn.Module):
    def __init__(self, in_channel, num_classes=10, out_channel=32, block_layers=[2, 4, 6, 8]): # num_classes=10 out_channel=32 4个dense_block中的conv_block的个数=[2,4,6,8]
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential( # 首先单卷积层和最大汇聚层
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        channels = 64 # 进入第一个dense_block的通道数为64
        block = []
        for i, layers in enumerate(block_layers): # 分别加入dense_block
            block.append(dense_block(channels, out_channel, layers))
            # 第一个dense_block: 3-->23 则第二个dense_block: 23-->x channels += layers * out_channel
            channels += layers * out_channel # 下一个dense_block的输入通道数（或接下来的transition_block的输入通道）为上一个dense_block的输出通道数
            if i != len(block_layers) - 1: # 添加transition_block
                block.append(transition_block(channels, channels // 2))  # 通过 transition 层将大小减半， 通道数减半
                channels = channels // 2
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        print(x.shape) # torch.Size([2, 3, 128, 128])
        x = self.block1(x)
        print(x.shape) # torch.Size([2, 64, 32, 32])
        x = self.block2(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


print("----------------------------------------------------------------")
X = torch.rand(4, 3, 128, 128)
model = DenseNet(in_channel=3)
summary(model,input_size=(3,128,128))
print(model)
print("----------------------------------------------------------------")
'''
输入DenseNet: torch.Size([2, 3, 128, 128]) 经起始卷积汇聚层: torch.Size([2, 64, 32, 32])
输入第一个dense_block: torch.Size([2, 64, 32, 32])
第一个dense_block输出: torch.Size([2, 128, 32, 32]) （out_channel=32 64 + 32 + 32 = 128）
输入第一个translation_block: torch.Size([2, 128, 32, 32])
第一个translation_block输出: torch.Size([2, 64, 16, 16]) 输入第二个dense_block: torch.Size([2, 64, 16, 16])
第二个dense_block输出: torch.Size([2, 192, 16, 16]) 输入第二个translation_block: torch.Size([2, 192, 16, 16])
第二个translation_block输出: torch.Size([2, 96, 8, 8]) 输入第三个dense_block: torch.Size([2, 96, 8, 8])
第三个dense_block输出: torch.Size([2, 288, 8, 8]) 输入第三个translation_block: torch.Size([2, 288, 8, 8])
第三个translation_block输出: torch.Size([2, 144, 4, 4]) 输入第四个dense_block: torch.Size([2, 144, 4, 4])
第四个dense_block输出: torch.Size([2, 400, 4, 4])

'''

def get_acc(output, label):
    total = output.shape[0]
    # output是概率，每行概率最高的就是预测值
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=128),
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(
    root='dataset/',
    train=True,
    download=True,
    transform=transform
)

# hand-out留出法划分
train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])

test_set = torchvision.datasets.CIFAR10(
    root='./dataset/',
    train=False,
    download=True,
    transform=transform
)

# train_loader = torch.utils.data.DataLoader(
#     dataset=train_set,
#     batch_size=batch_size,
#     shuffle=True
# )
# val_loader = torch.utils.data.DataLoader(
#     dataset=val_set,
#     batch_size=batch_size,
#     shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     dataset=test_set,
#     batch_size=batch_size,
#     shuffle=False
# )
#
# net = DenseNet(in_channel=3, num_classes=10)
# print(net)
#
# lr = 1e-2
# optimizer = optim.SGD(net.parameters(), lr=lr)
# critetion = nn.CrossEntropyLoss()
# net = net.to(device)
# prev_time = datetime.now()
# valid_data = val_loader
#
# for epoch in range(3):
#     train_loss = 0
#     train_acc = 0
#     net.train()
#
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         # forward
#         outputs = net(inputs)
#         loss = critetion(outputs, labels)
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         train_acc += get_acc(outputs, labels)
#         # 最后还要求平均的
#
#     # 显示时间
#     cur_time = datetime.now()
#     h, remainder = divmod((cur_time - prev_time).seconds, 3600)
#     m, s = divmod(remainder, 60)
#     # time_str = 'Time %02d:%02d:%02d'%(h,m,s)
#     time_str = 'Time %02d:%02d:%02d(from %02d/%02d/%02d %02d:%02d:%02d to %02d/%02d/%02d %02d:%02d:%02d)' % (
#         h, m, s, prev_time.year, prev_time.month, prev_time.day, prev_time.hour, prev_time.minute, prev_time.second,
#         cur_time.year, cur_time.month, cur_time.day, cur_time.hour, cur_time.minute, cur_time.second)
#     prev_time = cur_time
#
#     # validation
#     with torch.no_grad():
#         net.eval()
#         valid_loss = 0
#         valid_acc = 0
#         for inputs, labels in valid_data:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = net(inputs)
#             loss = critetion(outputs, labels)
#             valid_loss += loss.item()
#             valid_acc += get_acc(outputs, labels)
#
#     print("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f,"
#           % (epoch, train_loss / len(train_loader), train_acc / len(train_loader), valid_loss / len(valid_data),
#              valid_acc / len(valid_data))
#           + time_str)
#
#     torch.save(net.state_dict(), 'checkpoints/params.pkl')
#
# # 测试
# with torch.no_grad():
#     net.eval()
#     correct = 0
#     total = 0
#     for (images, labels) in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         output = net(images)
#         _, predicted = torch.max(output.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print("The accuracy of total {} val images: {}%".format(total, 100 * correct / total))
#
#
