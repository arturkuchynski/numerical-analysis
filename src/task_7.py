import matplotlib.pyplot as plt
import numpy as np


def lagrange_interpolation(x_interp_nodes, y_interp_nodes):
    def polynomial(arg):
        value = 0
        for i in range(0, len(y_interp_nodes)):
            value += y_interp_nodes[i] * polynomial_element(i, arg, x_interp_nodes)
        return value

    return polynomial


def polynomial_element(index, x, x_interp_nodes):
    val = 1
    for i in range(0, len(x_interp_nodes)):
        if i != index:
            val *= (x - x_interp_nodes[i]) / \
                   (x_interp_nodes[index] - x_interp_nodes[i])
    return val


def aitken_scheme(x_interp_nodes, y_interp_nodes, x_aitken):
    y = linear_interpolation(x_interp_nodes, y_interp_nodes, x_aitken)  # divided difference table
    return y[-1][0]


def linear_interpolation(x_interp_nodes, y_interp_nodes, x_aitken):

    f = [y_interp_nodes]
    for step in range(1, len(y_interp_nodes)):
        delta = list()
        for i in range(1, len(f[step - 1])):
            matrix = np.array([[f[step - 1][i - 1], x_interp_nodes[i - 1] - x_aitken],
                               [f[step - 1][i], x_interp_nodes[i + step - 1] - x_aitken]])
            polynom = 1 / (x_interp_nodes[i + step - 1] - x_interp_nodes[i - 1]) * np.linalg.det(matrix)
            delta.append(polynom)
        f.append(delta)

    return f


def display_plot(interpolation, x_interp_nodes, y_interp_nodes, x_aitken, y_aitken):
    start, end, step = 0.3, 0.8, 0.001

    x_range = np.arange(start, end, step)
    y_range = np.array([interpolation(x) for x in x_range])

    plt.plot(x_range, y_range, label='Lagrange polynomial')
    plt.plot(x_interp_nodes, y_interp_nodes, 'o', label='Interpolation nodes')
    plt.plot(x_aitken, y_aitken, 'o', label='Aitken method result')
    plt.legend(loc='best')
    plt.xlabel('x axes')
    plt.ylabel('y axes')
    plt.grid(True)
    plt.show()


def main():
    x_interp_nodes = [0.41, 0.46, 0.52, 0.60, 0.65, 0.72]
    y_interp_nodes = [2.57418, 2.32513, 2.09336, 1.86203, 1.74926, 1.62098]
    x_aitken = 0.487  # argument for aitken method

    """Lagrange polynomial"""
    y_aitken = aitken_scheme(x_interp_nodes, y_interp_nodes, x_aitken)
    interpolation = lagrange_interpolation(x_interp_nodes, y_interp_nodes)

    """Plot"""
    display_plot(interpolation, x_interp_nodes, y_interp_nodes, x_aitken, y_aitken)


if __name__ == '__main__':
    main()
