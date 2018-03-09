import math
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import random


def display_plot():
    x = np.arange(-10, 10, 1e-5)  # an array of arguments from -10 to 10 with step 10e-5
    func = np.sin(x**2) - 6*x + 1
    plt.plot(x, func, 'g')  # set curves and it's plot color
    plt.title('(sin(x^2)-6x+1')
    plt.axis([-10, 10, -60, 60])
    plt.grid(True)
    plt.show()


def f(x: float):
    return math.sin(x**2) - 6*x + 1


def method_of_chords(func, a, b, eps=10e-5):

    x = float(random.uniform(a, b))

    def derivative_f(x):
        return (func(x + eps) - func(x)) / (func(x + eps) - x)
    
    if derivative_f(x) > 0:
        def g(c):
            return a + func(a) / (func(c) - func(a)) * (c - a)
        x_n = b
    else:
        def g(c):
            return c - func(c) / (func(b) - func(c)) * (b - c)
        x_n = a

    iteration = 0

    # chords method step
    while math.fabs(func(x_n)) > eps:

        x_n = g(x_n)
        iteration += 1

    return x_n, iteration


def aitken_method(func, a, b, eps=10e-5):

    def derivative_f(x):
        return (func(x + eps) - func(x)) / (func(x + eps) - x)

    derivative_local_max = optimize.minimize_scalar(derivative_f, bounds=(a, b), method='bounded')

    # Let be g(x) = x - 1/M * func(x)
    # determine the value of g(x) depending on derivative's local maximum
    if derivative_local_max.fun > 0:
        def g(c):
            return c + (1.0 / derivative_local_max.x) * f(c)
    else:
        def g(c):
            return c - (1.0 / derivative_local_max.x) * f(c)
        
    # iterative step
    x_0 = float(random.uniform(a, b))
    x_1 = g(x_0)
    x_2 = g(x_1)
    x_temp = (x_0 * x_2 - x_1 ** 2) / (x_2 - 2 * x_1 + x_0)
    x_3 = g(x_temp)

    # iteration step
    iteration = 0
    while math.fabs(func(x_3)) > eps:
        x_0 = x_temp
        x_2 = g(x_1)
        x_1 = x_3

        x_temp = (x_0 * x_2 - x_1 ** 2) / (x_2 - 2 * x_1 + x_0)
        x_3 = g(x_temp)
        iteration += 1

    return x_3, iteration


def steffensen_method(func, a, b, eps=10e-5):

    x_n = float(random.uniform(a, b))

    def g(x):
        return x - (func(x) ** 2) / (func(x) - func(x - func(x)))

    # iteration step
    iteration = 0
    while math.fabs(func(x_n)) > eps:
        x_n = g(x_n)
        iteration += 1

    return x_n, iteration


def main():
    display_plot()
    print("###\nInput interval boundaries {}")
    lower, upper = float(input('left bound: ')), float(input('right bound: '))
    method_of_chords_res = method_of_chords(f, lower, upper)
    aitken_method_res = aitken_method(f, lower, upper)
    stephenson_method_res = steffensen_method(f, lower, upper)
    print("Method of chords:\n root {}, iterations {}".format(*method_of_chords_res))
    print("Aitken process:\n root {}, iterations {}".format(*aitken_method_res))
    print("Steffensen's method:\n root {}, iterations {}".format(*stephenson_method_res))


if __name__ == '__main__':
    main()
