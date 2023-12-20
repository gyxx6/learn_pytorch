import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# y = w1 * x^2 + w2 * x +b

w1 = torch.Tensor([1.0])
w1.requires_grad = True
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):  # 自动类型转换为张量间计算，取值需调用函数.item()
    return w1 * x * x + w2 * x + b


def loss(x, y):  # 自动构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict(before training):", 4, forward(4).item())

l_list = []
epoch_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 前向传播，构建计算图
        l.backward()  # 反向传播，构建计算图，自动求出所有参数梯度存入变量后，计算图自动释放
        print("\tgrad:", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())

        # w1.data = w1.data - 0.01 * w1.grad.item()也行
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()  # 权重数据清零
        w2.grad.data.zero_()
        b.grad.data.zero_()

        epoch_list.append(epoch)
        l_list.append(l.item())
    print("Epoch:", epoch, l.item())

print("predict(after training):", 4, forward(4).item())

plt.plot(epoch_list, l_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()