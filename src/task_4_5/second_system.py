import numpy as np


class SecondSystem:

    def f1(self, x, y):
        return np.sin(y + x) - 1.5 * x + 0.1

    def f2(self, x, y):
        return y ** 2 + x ** 2 - 1

    def df1dx(self, x, y):
        return np.cos(y + x) - 1.5

    def df1dy(self, x, y):
        return np.cos(y + x)

    def df2dx(self, x, y):
        return 2 * x

    def df2dy(self, x, y):
        return 2 * y

