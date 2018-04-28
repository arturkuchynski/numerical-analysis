import numpy as np
import matplotlib.pyplot as plt


def newton_interpolation_straight(x, y):

    # calculate divided difference
    f = divided_difference(x, y)

    def polynomial(arg):
        polynomial_i = y[0]
        delta = 1
        for i in range(1, len(f)):
            delta *= arg - x[i - 1]
            polynomial_i += f[i][0] * delta
        return polynomial_i

    return polynomial


def newton_interpolation_inverse(x, y):

    # calculate divided difference
    f = divided_difference(x, y)

    def polynomial(arg):
        polynom_val = y[-1]
        x_product = 1
        for i in range(1, len(f)):
            x_product *= arg - x[-i]
            polynom_val += f[i][-1] * x_product
        return polynom_val

    return polynomial


def divided_difference(x, y):

    f = [y]

    for step in range(1, len(y)):
        differences = list()
        for i in range(1, len(f[step - 1])):
            difference = (f[step - 1][i] - f[step - 1][i - 1]) / (x[i + step - 1] - x[i - 1])
            differences.append(difference)
        f.append(differences)

    return f


def display_plot(interpolation, x_interp_nodes, y_interp_nodes):

    # start, end, step = 0.000, 5.0001, 0.001
    start, end, step = 0.18, 0.225, 0.0005  # try
    # start, end, step = -4, 4, 0.001
    x_range = np.arange(start, end, step)
    y_range = np.array([interpolation(arg) for arg in x_range])

    # x_sec_range = [0.1857, 0.2165, 0.198, 0.2209, 0.1908, 0.2189]
    # y_sec_range = np.array([interpolation(arg) for arg in x_sec_range])

    # display plot of polynomial(x)
    plt.plot(x_range, y_range, label='Newton polynomial')
    # plt.plot(x_sec_range, y_sec_range, 'o', label='Additional points')

    # display interpolation nodes
    plt.plot(x_interp_nodes, y_interp_nodes, 'o', linewidth=0.00001,
             label='interpolation nodes')
    plt.legend(loc='best')
    plt.xlabel('x axes')
    plt.ylabel('y axes')
    plt.grid(True)
    plt.show()


def main():

    # vrt 11
    # x = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4,
    #      4.2, 4.4, 4.6, 4.8, 5]
    # y = [7.19, 7.466, 7.912, 8.495, 8.727, 8.842, 9.673, 9.622, 10.09, 10.744,
    #      10.988, 11.028, 11.897, 12.311, 12.737, 13.054, 13.166, 13.956, 13.949, 14.562, 15.312]


    x = [0.180, 0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215, 0.220, 0.225]
    y = [5.61543, 5.46547, 5.32159, 5.15326, 5.06478, 4.94856, 4.8317, 4.72545, 4.61855, 4.5182]

    ## y = x**2
    # x = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # y = [16, 9, 4, 1, 0, 1, 4, 9, 16]
    #
    # x = [0.0000, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
    #      4, 4.25, 4.5, 4.75, 5.0]
    # y = [4.588, 4.197, 4.129, 3.066, 3.111, 2.305, 2.348, 1.414, 0.926, 0.659,
    #      0.067, -0.818, -1.014, -1.389, -1.748, -2.8331, -2.854, -3.215, -3.981, -4.299, -4.843]

    str_interpolation = newton_interpolation_straight(x, y)
    display_plot(str_interpolation, x, y)

    inv_interpolation = newton_interpolation_inverse(x, y)
    display_plot(inv_interpolation, x, y)


if __name__ == '__main__':
    main()
