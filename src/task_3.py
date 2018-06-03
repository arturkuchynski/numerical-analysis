import matplotlib.pyplot as plt
import numpy as np


"""
    Laboratory task №3
    Variant 12, Artur Kuchynski
"""


def display_plot(func):
    x = np.arange(-10, 10, 10e-5)  # an array of arguments from -10 to 10 with step 10e-5
    plt.plot(x, func(x), 'g')  # set curve and it's plot color
    plt.title('P(x)= x^3-3*x^2-24*x-8')
    plt.axis([-7, 10, -30, 30])
    plt.grid(True)
    plt.show()


def f(x):
    return x**3-3*x**2-24*x-8  # polynomial function of power 3


def lobachevsky_method(func, coefficients, eps=10e-5):

    a0, a1, a2, a3 = coefficients
    roots = list()
    n = 1

    while True:

        k0 = a0**2
        k1 = a1**2 - 2*a0*a2
        k2 = a2**2 - 2*a1*a3
        k3 = a3**2

        x1 = (k1/k0)**(1/2**n)
        x2 = (k2/k1)**(1/2**n)
        x3 = (k3/k2)**(1/2**n)
        n += 1

        a0, a1, a2, a3 = k0, k1, k2, k3

        print("x={}\nx={}\nx={}\n######".format(x1, x2, x3))

        process_is_convergent = ((func(-x1) < eps and func(x1) < eps) and (
                                func(-x2) < eps and func(x2) < eps) and (func(-x3) < eps and func(x3) < eps))

        if process_is_convergent:  # break process
            for x_nodes in (x1, x2, x3):
                if -func(x_nodes) < eps:
                    roots.append(x_nodes)
                elif func(x_nodes) < eps:
                    roots.append(-x_nodes)
            return roots


def main():
    display_plot(f)
    coefficients = [1, -3, -24, -8, ]  # coefficients of the initial polynomial function
    result = lobachevsky_method(f, coefficients)
    print("\nLobachevsky method:\n")
    print("x1 = {}\nx2 = {}\nx3 = {}".format(*result))


if __name__ == '__main__':
    main()
