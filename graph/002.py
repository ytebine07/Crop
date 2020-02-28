import numpy as np
import matplotlib.pyplot as plt


def main():
    common = Common()

    file_path = "./data/set/positions.txt"
    # file_path = "./positions.txt"

    # grapth1(original)
    x, y = common.get_x_y(file_path)

    # graph2
    xx, yy = common.get_x_y(file_path)
    conv = Convolve(60, yy)
    yy = conv.calculate()

    # graph3
    xxx, yyy = common.get_x_y(file_path)
    conv = Convolve(30, yyy)
    yyy = conv.calculate()

    # calculate xticks
    xticks = []
    for i in range(len(x)//60):
        if i % 5 == 0:
            xticks.append(i)

    plt.figure(figsize=(20, 5), dpi=100)
    plt.xticks(xticks)

    plt.xlabel("seconds")
    plt.ylabel("center-position")

    plt.grid(axis='x', which="major")

    plt.plot(x, y, label='original', color="blue")
    plt.plot(xx, yy, label='average-60', color="orange")
    plt.plot(xxx, yyy, label='average-30', color="green")

    plt.legend()
    plt.savefig("./data/set/graph.png")


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from utils import Convolve, Common
    main()
