import numpy as np
import matplotlib.pyplot as plt


def get_x_y(path):
    x = []
    y = []
    with open(path) as f:
        for f_line in f:
            x.append(int(f_line.split(',')[0])/60)
            y.append(int(f_line.split(',')[1].strip()))

    return x, y


path_fastest = "./positions_fastest.txt"
x, y = get_x_y(path_fastest)

path_normal = "./positions_normal.txt"
xx, yy = get_x_y(path_normal)


plt.figure(figsize=(20, 5), dpi=100)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

plt.xlabel("seconds")
plt.ylabel("center-position")

plt.plot(x, y, label='fastest', color="blue")
plt.plot(xx, yy, label='normal', color="orange")

plt.legend()
plt.savefig("graph.png")
