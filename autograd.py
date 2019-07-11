import torch
torch.random.manual_seed(1)
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
# Each tensor has a .grad_fn attribute that references a Function that has created the Tensor
# (except for Tensors created by the user - their grad_fn is None).

print(y)
print(x.grad_fn)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)


# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. The input flag defaults to 'False' if not given
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))

print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)

# Let’s backprop now. Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.)).

out.backward(torch.tensor(1.))
print(x.grad)


x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2

while y.data.norm() < 1000:
    y = y * 2

print(y.grad_fn)
print(y)


v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)
print((x ** 2).grad_fn)

with torch.no_grad():
    print((x ** 2).requires_grad)