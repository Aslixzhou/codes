import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
tensor = torch.Tensor([[[1,2,3,4,5,6]]])
print(tensor,tensor.shape,tensor.type())
tensor = tensor.squeeze()
print(tensor,tensor.shape,tensor.type())
tensor = tensor.unsqueeze(0)
print(tensor,tensor.shape,tensor.type())





