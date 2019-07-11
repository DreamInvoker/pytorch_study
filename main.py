from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

print(x.size())
print(x.shape)

import numpy as np

print(np.zeros((3, 4)).size)
print(np.zeros((3, 4)).shape)

# Addition; syntax 1
y = torch.rand(5, 3)
print(y)
print(x + y)

# Addition: syntax 2
print(torch.add(x, y))

# Addition: syntax 3
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
y.add_(x)
print(y)


# Slice
print(x[:, 1])


# resize
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())


# get scalar tensor value
x = torch.randn(1)
print(x)
print(x.item())

# convert between numpy and Tensor
a = torch.ones(5)
print(a)
print(type(a), a.type())
b = a.numpy()
print(b, type(b))
a.add_(1)
print(a, b)


a = np.ones(5)
b = torch.from_numpy(a)
# np.add(a, 1, out=a)
a += 1
print(a, b)

# All the Tensors on the CPU except a CharTensor support converting to Numpy and back

print('torch.cuda.isavailable = ', torch.cuda.is_available())
x = torch.ones((2,3))
if torch.cuda.is_available:
    device = torch.device('cuda:2')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))

