import numpy as np


class Convolve:
    def __init__(self, average, input):
        """
        @args
            average : 移動平均を求めるフレーム数
            input   : ポジションのlist
        @see
            https://deepage.net/features/numpy-convolve.html
        @return
            移動平均後の整数の座標のlist
            原理上、元のリストより数が少なくなってしまう分は、最後の座標で保管
        """
        self._average = average
        self._input = input

    def calculate(self):

        v = np.ones(self._average)/self._average
        y3 = np.convolve(self._input, v, mode='valid')

        return_list = []
        frame_count = 0
        for i in self._input:
            if frame_count >= len(y3):
                # 足りない分は一番最後の座標を入れておく
                return_list.append(int(y3[len(y3)-1]))
            else:
                return_list.append(int(y3[frame_count]))
            frame_count += 1

        return return_list

    def hoge(self):
        return "hogeeee"
