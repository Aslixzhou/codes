import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torchvision.datasets as dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
train_data = dataset.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
test_data = dataset.MNIST(root="./dataset",train=False,transform=transforms.ToTensor(),download=False)
import torch.utils.data as data_utils
train_loader = data_utils.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data,batch_size=64,shuffle=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=5,padding=2), # 输入通道1 输出28*28*32
            torch.nn.BatchNorm2d(32), # 指定通道数
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2) # 输出14*14*32
        )
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = torch.nn.Linear(14*14*32,10) # 全连接层
    def forward(self,x):
        out = self.conv(x)
        out = self.flatten(out)
        # out = out.view(out.size()[0],-1) # 展平为一维向量
        out = self.fc(out)
        return out

cnn = CNN()
batch_size=64
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=0.01)
for epoch in range(10):
    train_loss = 0
    correct = 0
    for i,(images,labels) in enumerate(train_loader):
        images = images
        labels = labels
        outputs = cnn(images)
        loss = loss_func(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = outputs.max(1)
        train_loss += loss.item()
        correct += (pred == labels).sum().item()
    train_loss /= (len(train_data)//batch_size)
    print("epoch is {}, ite is {}/{}, train_loss is {}, accuracy is {}".format(epoch+1,i,len(train_data)//batch_size,train_loss,correct/len(train_data)))

    loss_test = 0
    accuracy = 0
    for i,(images,labels) in enumerate(test_loader):
        images = images
        labels = labels
        outputs = cnn(images)
        loss_test += loss_func(outputs,labels)
        _,pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()
    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data)//batch_size)
    print("epoch is {}, accuracy is {}, loss_test is {}.".format(epoch+1,accuracy,loss_test.item()))
