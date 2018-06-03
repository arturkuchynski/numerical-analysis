import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def approximate_by_linear(x_nodes, y_nodes):
    system = np.array([[sum(x ** 2 for x in x_nodes), sum(x_nodes)],
                       [sum(x_nodes), len(x_nodes)]])
    s = np.array([sum(x_nodes[i] * y_nodes[i] for i in range(len(x_nodes))),
                  sum(y_nodes)])

    a, b = np.linalg.solve(system, s)

    def approximation(x):
        return a * x + b

    return approximation


def approximate_by_quadratic(x_nodes, y_nodes):
    system = np.array([[sum(x ** 4 for x in x_nodes), sum(x ** 3 for x in x_nodes), sum(x ** 2 for x in x_nodes)],
                       [sum(x ** 3 for x in x_nodes), sum(x ** 2 for x in x_nodes), sum(x_nodes)],
                       [sum(x ** 2 for x in x_nodes), sum(x_nodes), len(x_nodes)]])
    s = np.array([sum((x_nodes[i] ** 2) * y_nodes[i] for i in range(0, len(x_nodes))),
                  sum(x_nodes[i] * y_nodes[i] for i in range(0, len(x_nodes))),
                  sum(y_nodes)])

    a, b, c = np.linalg.solve(system, s)

    def approximation(x):
        return a * (x ** 2) + b * x + c

    return approximation


def approximate_by_exponential(x_nodes, y_nodes):
    system = np.array([[sum(x ** 2 for x in x_nodes), sum(x_nodes)],
                       [sum(x_nodes), len(x_nodes)]])
    s = np.array([sum(x_nodes[i] * np.log(y_nodes[i]) for i in range(len(x_nodes))),
                  sum(np.log(y) for y in y_nodes)])

    a, b = np.linalg.solve(system, s)

    def approximation(x):
        return a * np.exp(np.exp(b) * x)

    return approximation


def approximate_by_logarithmic(x_nodes, y_nodes):
    system = np.array([[sum(np.log(x) ** 2 for x in x_nodes), sum(np.log(x) for x in x_nodes)],
                       [sum(np.log(x) for x in x_nodes), len(x_nodes)]])
    s = np.array([sum(y_nodes[i] * np.log(x_nodes[i]) for i in range(len(x_nodes))),
                  sum(y_nodes)])

    a, b = np.linalg.solve(system, s)

    def approximation(x):
        return a * np.log(x) + b

    return approximation


def approximate_by_hyperbolic(x_nodes, y_nodes):
    system = np.array([[sum(1.0 / (x ** 2) for x in x_nodes), sum(1.0 / x for x in x_nodes)],
                       [sum(1.0 / x for x in x_nodes), len(x_nodes)]])
    s = np.array([sum(y_nodes[i] / x_nodes[i] for i in range(len(x_nodes))),
                  sum(y_nodes)])
    a, b = np.linalg.solve(system, s)

    def approximation(x):
        return a / x + b

    return approximation


def display_plot(x_nodes, y_nodes):
    x_range = np.arange(1.00, 6.5, 0.0001)
    # x_range = np.arange(0.43, 4.7, 0.0001)

    plt.plot(x_nodes, y_nodes, 'o', label='Points M(i)')

    linear = approximate_by_linear(x_nodes, y_nodes)
    y_range = np.array([linear(x) for x in x_range])
    plt.plot(x_range, y_range, label='Linear approximation')

    quadratic = approximate_by_quadratic(x_nodes, y_nodes)
    y_range = np.array([quadratic(x) for x in x_range])
    plt.plot(x_range, y_range, label='Quadratic approximation')

    exponential = approximate_by_exponential(x_nodes, y_nodes)
    y_range = np.array([exponential(x) for x in x_range])
    plt.plot(x_range, y_range, label='Exponential approximation')

    logarithmic = approximate_by_logarithmic(x_nodes, y_nodes)
    y_range = np.array([logarithmic(x) for x in x_range])
    plt.plot(x_range, y_range, label='Log approximation')

    hyperbolic = approximate_by_hyperbolic(x_nodes, y_nodes)
    y_range = np.array([hyperbolic(x) for x in x_range])
    plt.plot(x_range, y_range, label='Hyperbolic approximation')

    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x axes')
    plt.ylabel('y axes')
    plt.show()


def main():
    # Variant 12
    x_nodes = [1.16, 1.88, 2.60, 3.32, 4.04, 4.76, 5.48, 6.20]
    y_nodes = [0.18, 0.26, 0.32, 0.36, 0.40, 0.43, 0.95, 0.85]

    # Variant 19
    # x_nodes = [0.41, 0.97, 1.53, -2.09, 2.65, 3.21, 3.77, 4.33]
    # y_nodes = [0.45, 1.17, 1.56, 1.82, 2.02, 2.18, 2.31, 2.44]

    display_plot(x_nodes, y_nodes)


if __name__ == '__main__':
    main()
