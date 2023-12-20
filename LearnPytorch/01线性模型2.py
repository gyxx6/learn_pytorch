import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# y=w*x+b
# 此处设为 y = 2 * x + 3
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 7.0, 9.0]


def forward(x):
    return w * x + b


def loss(x, y):  # MSE
    y_pred = forward(x)
    return (y - y_pred) ** 2


mse_list = []  # 对应w权重的MSE
W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 6.1, 0.1)
[w, b] = np.meshgrid(W, B)  # [X,Y]=np.meshgrid(x,y) 函数用两个坐标轴上的点在平面上画网格。

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pre_val = forward(x_val)
    print(y_pre_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))  # Axes3D是mpl_toolkits.mplot3d中的一个绘图函数
ax.plot_surface(w, b, l_sum / 3)  # 画曲面图---Axes3D.plot_surface(X, Y, Z)
ax.set_xlabel("W")
ax.set_ylabel("B")
ax.set_zlabel("Cost Value")
plt.show()