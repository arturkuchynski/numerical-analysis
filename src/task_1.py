import math
import random
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile


@profile()
def display_plot():
    x = np.arange(-10, 10, 1e-5)  # making an array of arguments from -10 to 10 with step 10e-5
    func = (1.0 / np.tan(x)) - ((3.0 * x) / 4.0)
    plt.plot(x, func, 'g')  # set curves and it's color for plot
    plt.title('(1/ctg(x)) - (3/4)*x = 0')
    plt.axis([-3, 4, -10, 10])
    plt.grid(True)
    plt.show()


def f(x):
    if math.tan(x) == 0:
        return float('Inf') - ((3.0 * x) / 4.0)
    else:
        return (1.0 / math.tan(x)) - ((3.0 * x) / 4.0)


def bisection_method(func, a, b, eps):
    x = (a + b) / 2.0
    iteration = 0

    while math.fabs(func(x)) > eps:  # step

        x = (a + b) / 2.0

        if (func(x) < 0 and func(a) < 0) or (func(x) > 0 and func(a) > 0):
            a = x
        else:
            b = x

        iteration += 1

    return x, func(x), iteration


def newtons_method(func, x, eps):
    delta_x = math.fabs(func(x))
    iteration = 0

    # calculate derivative of the function according to i'ts definition
    def derivative_f(x):
        return (func(x + eps) - func(x)) / ((x + eps) - x)

    while delta_x > eps:
        x = x - func(x) / derivative_f(x)
        delta_x = math.fabs(func(x))
        iteration += 1

    return x, func(x), iteration


def simple_iteration_method(func, a, b, eps):
    # determine derivative of the function and it's local maximum at the segment [a,b]
    # try also scipy.misc.derivative(func, lower_bound, upper_bound, order_of_derivative)
    def derivative_f(x):
        return (func(x + eps) - func(x)) / (func(x + eps) - x)

    local_max_f = optimize.minimize_scalar(derivative_f, bounds=(a, b), method='bounded')

    # determine the value of g(x) depending on derivative's local maximum
    if local_max_f.fun > 0:
        def g(c): return c + (1.0 / local_max_f.x) * f(c)
    else:
        def g(c): return c - (1.0 / local_max_f.x) * f(c)

    iteration = 0

    # select x from the segment
    x = float(random.uniform(a, b))

    # sufficient condition for convergence
    if a <= g(x) <= b:
        # iterative step
        while math.fabs(func(x)) > eps:
            x = g(x)
            iteration += 1

        return x, func(x), iteration  # OK
    else:
        return float('Inf')  # iterative process is divergent


def main():
    display_plot()
    for n in ('x1', 'x2', 'x3'):
        print("###\nFor {}".format(n))
        lower, upper = float(input('lower bound: ')), float(input('upper bound: '))
        approx_root = float(input('Initial approximation of root: '))  # good enough [-0.88, 1, 3.5]
        bisection_res = bisection_method(f, lower, upper, 10e-6)
        newton_res = newtons_method(f, approx_root, 10e-5)
        iteration_res = simple_iteration_method(f, lower, upper, 10e-5)
        print("###\nBisection method for {}".format(n))
        print("Root is : {}\nf(x) at root is: {} iterations: {}\n###".format(*bisection_res))
        print("Newton's method for {}".format(n))
        print("Root is : {}\nf(x) at root is: {} iterations: {}\n###".format(*newton_res))
        print("Simple iteration method for {}".format(n))
        if iteration_res == float('Inf'):
            print("Iterative process is divergent")
        else:
            print("Root is : {}\nf(x) at root is: {} iterations: {}\n###".format(*iteration_res))


if __name__ == '__main__':
    main()
