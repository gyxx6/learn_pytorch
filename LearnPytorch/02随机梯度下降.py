import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]
w = 1.0


def forward(x):
    return x * w


def loss(x, y):  # 区别普通梯度下降：损失函数未计算MSE均值
    y_pred = forward(x)
    return (y - y_pred) ** 2


def SGD(x, y):  # 区别普通梯度下降
    return 2 * x * (w * x - y)


print('Predict(before training)', 4, forward(4))

epoch_list = []
loss_list = []

for epoch in range(100):  # 下左图训练100轮次，右图训练10轮次
    for x, y in zip(x_data, y_data):  # 区别普通梯度下降
        grad = SGD(x, y)
        w -= 0.01 * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
        epoch_list.append(epoch)
        loss_list.append(l)

print('Predict(after training)', 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()