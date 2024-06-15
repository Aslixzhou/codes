import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import  nn

class SEnet(nn.Module):
    def __init__(self,in_channel,ratio=16): # 输入通道数 缩放比例
        super(SEnet,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 高宽全局平均池化 输出高宽为1
        # 两次全连接
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//ratio,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//ratio,out_features=in_channel,bias=False),
            nn.Sigmoid(),
        )

    def forward(self,x):
        b,c,h,w = x.size()
        print(x,x.shape)
        # b,c,h,w --> b,c,1,1 --> b,c
        out = self.avg_pool(x).view([b,c])
        # b,c --> b,c//ratio --> b,c --> b,c,1,1
        out = self.fc(out).view([b,c,1,1])
        print(out,out.shape)
        print(x*out,(x*out).shape)
        return x * out

model = SEnet(in_channel=512)
print(model)
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)


