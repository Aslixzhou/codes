import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import  nn

# 通道注意力机制
class Channel_attention(nn.Module):
    def __init__(self,in_channel,ratio=16): # 输入通道数 缩放比例
        super(Channel_attention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 高宽全局平均池化 输出高宽为1
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 高宽全局最大池化 输出高宽为1
        # 两次全连接
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//ratio,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//ratio,out_features=in_channel,bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        out1 = self.avg_pool(x).view([b,c])
        out2 = self.max_pool(x).view([b,c])
        out1 = self.fc(out1)
        out2 = self.fc(out2)
        out = out1 + out2
        out = self.sigmoid(out).view([b,c,1,1]) # 获得各个特征层的比重
        return x * out


# 空间注意力机制
class Spatial_attention(nn.Module):
    def __init__(self,kernel_size=7): # 卷积核大小
        super(Spatial_attention,self).__init__()
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=False) # , 多了一个逗号！
        '''
        self.conv的那一行末尾的逗号,。在Python中，如果在一行的末尾使用逗号，它会创建一个包含一个元素的元组，而不是单独的对象。因此，self.conv最终会成为一个包含nn.Conv2d对象的元组，而不是单独的nn.Conv2d对象。
        '''
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool_out, _ = torch.max(x,dim=1,keepdim=True)
        print(max_pool_out.shape)
        mean_pool_out = torch.mean(x,dim=1,keepdim=True)
        print(mean_pool_out.shape)
        pool_out = torch.cat([max_pool_out,mean_pool_out],dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return x * out

class Cbam(nn.Module):
    def __init__(self,in_channel,ratio=16,kernel_size=7):
        super(Cbam, self).__init__()
        self.channel_attention = Channel_attention(in_channel=in_channel,ratio=ratio)
        self.spatial_attention = Spatial_attention(kernel_size=kernel_size)

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

model = Cbam(512)
print(model)
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)

