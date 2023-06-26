import numpy as np
import xlrd
import matplotlib.pyplot as plt

from matplotlib import animation

layer_sample = np.arange(0, 6, 1) # total 6
width_sample = np.arange(4, 50, 2) # total 23
layer, width = np.meshgrid(layer_sample, width_sample)

wb_1 = xlrd.open_workbook('./2d_offline.xls')
sheet_1 = wb_1.sheet_by_index(0)
value = np.zeros((23, 6))
for i in range(23):
    for j in range(6):
        value[i][j] = sheet_1.cell_value(layer[i][j], int((width[i][j] - 4) / 2))

fig = plt.figure(figsize=(10, 8))
ax_1 = fig.add_subplot(111, projection='3d')
surf = ax_1.plot_surface(layer, width, value, color='b', alpha=0.6, label='ResNeXt')
# plt.show()

wb_2 = xlrd.open_workbook('./trial1_pi.xls')
sheet_2 = wb_2.sheet_by_index(0)
history = np.zeros((33, 3))
for i in range(3):
    for j in range(3):
        history[i][j] = sheet_2.cell_value(i, j)


def init():
    pass


def animate(i):
    ax_1.clear()
    ax_1.plot_surface(layer, width, value, color='b', alpha=0.7, label='ResNeXt')
    for j in range(3):
        history[i + 3][j] = sheet_2.cell_value(i + 3, j)
    ax_1.scatter(history[:(i + 3), 0], history[:(i + 3), 1], history[:(i + 3), 2],
                 s=30, color='green', label='observations')
    ax_1.scatter(history[(i + 3):(i + 4), 0], history[(i + 3):(i + 4), 1], history[(i + 3):(i + 4), 2],
                 s=20, color='red', label='new observation')
    for k in range(i + 3):
        ax_1.text(history[k][0], history[k][1], history[k][2], str(k + 1),
                  color='black', fontsize="medium", fontweight='bold')
    ax_1.set_title('ResNext. Step: %s' % str(i))


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=1000, blit=False)
anim.save('resnext_2d_cifar_pi.gif', writer='pillow', fps=1, dpi=300)