import torch
import matplotlib.pyplot as plt

# Pytorch中计算图是mini-batch类型(样本结果一次性求出)，X,Y均为3*1张量
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 将模型定义为类。模型类应该从nn.Module继承，这是对于所有神经网络模型的基本类(所有神经网络的模板)
class LinearModel(torch.nn.Module):  # 以下两个函数必须都有且如此命名
    def __init__(self):
        super(LinearModel, self).__init__()  # 调用父类构造函数
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，可计算x*w+b。.(1,1):(输入特征维数,输出特征维数).

    def forward(self, x):  # 前向计算函数，overwrite父类__call()__（此方法可以使类像函数一样被调用）
        y_pred = self.linear(x)  # 可调用的对象
        return y_pred
    # 不用定义后向传播函数：Moduel构造对象自动根据计算图实现后向


model = LinearModel()  # 可直接将x送进去model(x)然后被调用求出y^

# 构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)  # 不求均值，默认求均值。求损失，需要y、y_pred

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD是类，需要实例化，第一个参数是指明需要优化的参数。.parameters()可将模型中需要训练的参数全都找出

epoch_list = []
l_list = []

for epoch in range(100):

    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)  # loss是一个对象，打印时会自动调用__str__()，不会产生计算图

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # 根据梯度和学习率自动更新参数

    epoch_list.append(epoch)
    l_list.append(loss.item())

# 输出w、b
print('w=', model.linear.weight.item())  # .item()显示[[]]中的数值
print('b=', model.linear.bias.item())

# 测试模型
x_test = torch.Tensor([[4.0]])  # 1x1矩阵
y_test = model(x_test)
print('y_pred = ', y_test.data)

plt.plot(epoch_list,l_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()