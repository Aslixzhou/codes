import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn,optim
from  torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# 残差块
class ResBlook(nn.Module):
    def __init__(self,ch_in,ch_out,stride = 1): # 输入通道 输出通道 步长值 根据图结构设置参数ch_in,ch_out
        super(ResBlook, self).__init__()
        # 卷积层C1
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        # batch normaliziton B1:对卷积层输出进行标准正态化
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 卷积层C2
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        # batch normaliztion B2:对卷积层输出进行标准正态化
        self.bn2 = nn.BatchNorm2d(ch_out)
        # 如果已经相等了，就不需要再使用这个额外操作了,因此输出一个空的Sequential
        self.extra = nn.Sequential()
        # 想要保持shape相等，做一次kernel_size = 1,stride = 1的卷积即可
        if ch_out != ch_in: # 通道数改变 经过1*1卷积调整
            self.extra = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    #构造前向过程，这里不仅要注意shape还要注意添加短路层
    def forward(self,x):
        # x:[b,ch,h,w]
        # x 经过卷积层C1再BN1 经过卷积层C2再BN2,最后一层可以不需要relu
        # 可以根据自己的设计多加几层（模仿网络结构或添加结构）
        # 这里输出我们使用out(不是x),因为后面要做短路层
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 短路(short cut)
        # x:[b,ch_in,h,w] add with out:[b,ch_out,h,w]
        # 这里要注意ch_in 和ch_out 不相等不能运行，需要对输出做个额外的操作
        # extra:[b,ch_in,h,w]=>[b,ch_out,h,w]
        out = self.extra(x) + out
        return out

# 构造一个18层的resnet类
class ResNet18(nn.Module):
        def __init__(self):
            super(ResNet18, self).__init__()
            # 卷积层C1+BN2
            self.conv1 = nn.Sequential(
                # channel3转化为channel为64
                nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64)
            )
            # 再添加四个resblock残差短路块【测试后调整stride】
            # x:[b,3,h,w] => [b,64,h,w] ==> [b,128,h,w]
            self.blk1 = ResBlook(64,128,stride=2)
            # [b,128,h,w] => [b,256,h,w]
            self.blk2 = ResBlook(128,256,stride=2)
            #[b,256,h,w] => [b,512,h,w]
            self.blk3 = ResBlook(256,512,stride=2)
            self.blk4 = ResBlook(512, 512)
            # self.outlayer = nn.Linear(?,10) 测试确定 ?
            self.outlayer = nn.Linear(2048, 10)

        # 构造forward过程
        def forward(self,x):
            # [b,3,32,32] = [b,64,32,32]
            x = F.relu(self.conv1(x))
            # 经过4个单元的残差块.[b,64,32,32] => [b,1024,2,2]
            x = self.blk1(x)
            x = self.blk2(x)
            x = self.blk3(x)
            x = self.blk4(x)
            # [2,3,32,32] => [2, 512, 4, 4]
            # print(x.shape) # torch.Size([2, 512, 4, 4])
            # 平均池化层
            x = F.adaptive_avg_pool2d(x,[2,2])
            # print(x.shape) # torch.Size([2, 512, 2, 2])
            # [2,512,1,1] => [2,512*1*1]
            x = x.view(x.size(0),-1)
            # print(x.shape) # torch.Size([2, 2048])
            x = self.outlayer(x)
            return x

def model_test():
    x = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(x)
    print("resnet: ",out.shape)
    print(model)

def model_main():
    batchsize = 128
    # 在当前文件夹下新建一个cifar文件夹cifar，加载cifar10训练集，对数据进行一些变换
    cifar_train = datasets.CIFAR10(
        './dataset',
        train=True,
        transform=transforms.Compose([
                # 将照片转化成32*32的输出特征图
                transforms.Resize((32,32)),
                # 转化为tensor 暂时不做normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]),
        download=True,
    )

    # 选择前 4096 个样本
    subset_indices = range(4096)
    cifar_train_subset = Subset(cifar_train, subset_indices)

    cifar_train = DataLoader(cifar_train_subset,batch_size=batchsize,shuffle=True)

    cifar_test = datasets.CIFAR10(
        './dataset',
        train=False,
        transform=transforms.Compose([
            # 将照片转化成32*32的输出特征图
            transforms.Resize((32, 32)),
            # 转化为tensor 暂时不做normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ]),
        download=True
    )
    # 选择前 4096 个样本
    subset_indices = range(1024)
    cifar_test_subset = Subset(cifar_test, subset_indices)
    cifar_test = DataLoader(cifar_test_subset, batch_size=batchsize, shuffle=True)

    # 用iter方法得到cifar_train的迭代器，并用next()方法生成数据
    x,label = next(iter(cifar_train))
    print('x:',x.shape,'label:',label.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = ResNet18()
    model = resnet.to(device)
    print(model)

    # 构建criteon,导入nn，使用CrossEntropyLoss,并转入cuda加速
    criteon = nn.CrossEntropyLoss().to(device)
    # 构建优化器，可以改成SGD等其它优化器，参数自动调用，learning_rate设置为1e-3
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # 迭代2次
    epochs = 2
    for epoch in range(epochs):
        model.train()
        sum = 0
        train_loss = 0.0
        train_corrects = 0
        total_samples = 0
        # 遍历训练集，使用enumerate调取当前的batch的id，图片数据x，以及对应的标签label
        for batch_idx,(x, label) in enumerate(cifar_train):
            sum += 128
            x, label = x.to(device),label.to(device)
            logits = model(x) # logits:[b,10] label:[b] loss 是一个tensor标量
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            train_corrects += torch.sum(preds == label.data)
            total_samples += x.size(0)
            # print(f'{sum} / {len(cifar_train_subset)}')

        train_loss = train_loss / total_samples
        train_accuracy = train_corrects.double() / total_samples
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x,label in cifar_test:
                x,label = x.to(device),label.to(device)
                logits = model(x)
                pred = logits.argmax(dim = 1)
                # 与label做比较，统计分类正确的数量（eq返回等shape的0-1向量）
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
            acc = total_correct / total_num
        print(epoch, 'TEST ACC:', acc) # 做一些最后的改进


if __name__ == '__main__':
    model_test()
    model_main()

