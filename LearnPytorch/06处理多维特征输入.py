import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 同目录下需要有diabetes.csv文件，分隔符用逗号，（非英伟达特斯拉等高级显卡）数据类型用float
xy = np.loadtxt('diabetes.csv.gz',delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])#读取所有行，从第一列开始最后一列不要（读前8列x标签）
y_data = torch.from_numpy(xy[:, [-1]]) #读取所有行的最后一列，加中括号为生成矩阵


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 运算模块，作为一层，无参数不需要训练

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True,reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

loss_list = []
epoch_list = []

for epoch in range(10000):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item(),y_pred)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    loss_list.append(loss.item())  # 应该将标量加入数组，否则为Tensor形式
    epoch_list.append(epoch)

    y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

    acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
    print(epoch," loss=", loss.item(), " acc=", acc)


# print(epoch_list)
# print(loss_list)
plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()