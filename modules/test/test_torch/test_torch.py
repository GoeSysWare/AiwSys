from __future__ import print_function
import torch


# x = torch.empty(5, 3)
# print(x)

# device = torch.device("cuda")          # a CUDA device object
# y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
# x = x.to(device)                       # 或者使用`.to("cuda")`方法
# z = x + y
# print(z)
# print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype


x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)


z = y * y * 3
out = z.mean()

print(z, out)

out.backward()

print(x.grad)


x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)