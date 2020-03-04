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


path_fastest = "./data/set2/positions.txt"
x, y = get_x_y(path_fastest)

path_normal = "./original_positions.txt"
xx, yy = get_x_y(path_normal)

# calculate xticks
xticks = []
for i in range(len(x)//60):
    if i % 5 == 0:
        xticks.append(i)

plt.figure(figsize=(20, 5), dpi=100)
plt.xticks(xticks)

plt.xlabel("seconds")
plt.ylabel("center-position")

plt.plot(x, y, label='local', color="blue")
plt.plot(xx, yy, label='colab', color="orange")

plt.legend()
plt.savefig("graph2.png")
