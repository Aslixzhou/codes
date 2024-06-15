import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn
import math

class Eca_block(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1): # 根据in_channel计算卷积核大小
        super(Eca_block,self).__init__()
        kernel_size = int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.conv = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False) # 1d卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avg_pool(x).view([b,1,c]) # Conv1d 三维张量 [batch_size,in_channels,sqeuence_length]
        out = self.conv(avg)
        out = self.sigmoid(out).view([b,c,1,1])
        return x * out


model = Eca_block(512)
print(model)
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)

