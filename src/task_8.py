import numpy as np
import bisect
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
# import scipy.misc


def parabolic_spline(x_interp_nodes, y_interp_nodes):
    a = y_interp_nodes  # Note that a[i] == y[i], i = 1,len(y_nodes)

    h = [0] * (len(y_interp_nodes) - 1)
    for i in range(len(y_interp_nodes) - 1):
        h[i] = (x_interp_nodes[i + 1] - x_interp_nodes[i])

    z = [2 * (y_interp_nodes[i + 1] - y_interp_nodes[i]) / h[i]
         for i in range(len(y_interp_nodes) - 1)]

    b = [0] * (len(y_interp_nodes))  # init empty list b

    b[0] = a[0]  # set border conditions for derivatives

    for i in range(len(y_interp_nodes)):
        b[i] = z[i - 1] - b[i - 1]

    c = [0] * (len(y_interp_nodes) - 1)
    for i in range(len(y_interp_nodes) - 1):
        c[i] = (b[i + 1] - b[i]) / (2 * h[i])

    def interpolate(x):
        # insert x into x_i sorted list and get i position
        i = bisect.bisect(x_interp_nodes, x)

        return a[i - 1] + (x - x_interp_nodes[i - 1]) * b[i - 1] + \
            + c[i - 1] * (x - x_interp_nodes[i - 1]) ** 2

        # def s(x):
        #     return a[i - 1] + (x - x_interp_nodes[i - 1]) * b[i - 1] + \
        #         + c[i - 1] * (x - x_interp_nodes[i - 1]) ** 2
        #
        # if scipy.misc.derivative(s, x) == x:
        #     return s(x)
        # else:
        #     return a[i-1]

    return interpolate


def display_plot(x_interp_nodes, y_interp_nodes):
    x_range = np.arange(-5, 4, 0.01)
    quadratic_polynomial = parabolic_spline(x_interp_nodes, y_interp_nodes)
    y_range = interp1d(x_interp_nodes, y_interp_nodes, kind='quadratic')
    cubic_polynomial = CubicSpline(x_interp_nodes, y_interp_nodes)
    plt.plot(x_interp_nodes, y_interp_nodes, 'o', label='Interpolation nodes')
    plt.plot(x_range, [quadratic_polynomial(x) for x in x_range], label='Parabolic spline')
    plt.plot(x_range, cubic_polynomial(x_range), label='Cubic spline')
    plt.plot(x_range, [y_range(x) for x in x_range], label='Parabolic spline from scipy')
    plt.legend(loc='best')
    plt.xlabel('x axes')
    plt.ylabel('y axes')
    plt.grid(True)
    plt.show()


def main():
    x_interp_nodes = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_interp_nodes = [12.5, 7.8, 2.3, 0.4, -4.1, 0.2, 1.9, 4.8, 9.4, 10.5]
    display_plot(x_interp_nodes, y_interp_nodes)


if __name__ == '__main__':
    main()
