import numpy as np
import matplotlib.pyplot as plt


def get_x_y(path):
    x = []
    y = []
    with open(path) as f:
        for f_line in f:
            x.append(f_line.split(',')[0])
            y.append(int(f_line.split(',')[1].strip()))

    return x, y


path_fastest = "./positions_fastest.txt"
x, y = get_x_y(path_fastest)

path_normal = "./positions_normal.txt"
xx, yy = get_x_y(path_normal)


plt.figure(figsize=(20, 5), dpi=100)
plt.xticks([0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000])

plt.plot(x, y, label='fastest', color="blue")
plt.plot(xx, yy, label='normal', color="orange")

plt.legend()
plt.savefig("graph.png")
