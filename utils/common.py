class Common:

    def get_x_y(self, path):
        """ファイルパスを渡してフレームとポジションのリストを返却する

            flame,position のファイルのみに対応
            flameは60で割って秒相当にして返却

        """
        x = []
        y = []
        with open(path) as f:
            for f_line in f:
                x.append(int(f_line.split(',')[0])/60)
                y.append(int(f_line.split(',')[1].strip()))

        return x, y
