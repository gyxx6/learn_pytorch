import numpy as np
import torch
from torch.utils.data import Dataset  # Dataset抽象类，不能实例化，只能被其他子类继承
from torch.utils.data import DataLoader


# 1.Prepars data
class DiabetesDataset(Dataset):
    def __init__(self, filepath):  # 1.所有数据读入内存 2.数据较大，文件名放入列表，后用getitem()方法通过索引读入数据
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # 返回(N,9)，取出N
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  # 魔法方法，支持通过下标(索引)取数的操作
        return self.x_data[index], self.y_data[index]  # 返回元祖(x,y);分开训练更方便

    def __len__(self):  # 魔法方法，可返回数据条数
        return self.len


dataset = DiabetesDataset('diabetes.csv')

train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # 读入数据时的并行线程数
# print(train_loader)


# 2.
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':  # Windows和Linux下处理多进程的库不同，添加避免报错
    for epoch in range(100):  # MiniBatch需要的双重循环
        for i, data in enumerate(train_loader, 0):  # enumerate()获得当前迭代次数。train_loader中拿出的(x,y)元组放入data中
            # 1.Prepare data
            inputs, labels = data  # data拿出x[i]、y[i]放入inputs、labels时自动变成Tensor

            # 2.Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            # 3.Backward
            optimizer.zero_grad()
            loss.backward()

            # 4.Update
            optimizer.step()
