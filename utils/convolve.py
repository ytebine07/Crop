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
        """
        self._average = average
        self._input = input

    def calculate(self):

        v = np.ones(self._average)/self._average
        y3 = np.convolve(self._input, v, mode='same')

        # 最初と最後の計算が上手くできないので、
        # オリジナルの座標を入れておく
        return_list = []
        frame_count = 0
        for i in self._input:
            if frame_count <= self._average:
                # 動画冒頭
                return_list.append(self._input[frame_count])
            elif len(self._input) - frame_count <= self._average:
                # 動画最後
                return_list.append(self._input[frame_count])
            else:
                return_list.append(int(y3[frame_count]))
            frame_count += 1

        return return_list

    def hoge(self):
        return "hogeeee"
