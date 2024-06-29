import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
tensor = torch.Tensor([[[1,2,3,4,5,6]]])
print(tensor,tensor.shape,tensor.type())
tensor = tensor.squeeze()
print(tensor,tensor.shape,tensor.type())
tensor = tensor.unsqueeze(0)
print(tensor,tensor.shape,tensor.type())



import torch
import torch.nn.functional as F

target = torch.tensor([3])
num_classes = 1000  # 假设类别总数为 1000，您需要根据实际情况修改这个值

one_hot_target = F.one_hot(target, num_classes=num_classes)
print(one_hot_target,one_hot_target.shape)

