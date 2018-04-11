import numpy as np


class FirstSystem:

    def f1(self, x, y):
        return np.cos(x - 1) + y - 0.5

    def f2(self, x, y):
        return x - np.cos(y) - 3

    def df1dy(self, x, y):
        return 1

    def df1dx(self, x, y):
        return -np.sin(x-1)

    def df2dy(self, x, y):
        return np.sin(y)

    def df2dx(self, x, y):
        return 1
    #
    # def explicit_f1(self, x):
    #
    #     return np.arccos(- x + 0.5) + 1
    #
    # def explicit_f2(self, x):
    #
    #     return np.cos(x) + 3
    #

