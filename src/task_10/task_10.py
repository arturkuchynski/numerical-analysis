import numpy as np


def rectangle_method(f, a, b, div=5, method='left', h=None):

    if h is None:
        h = (b - a) / div
    else:
        div = int((b - a) / h)

    s = 0
    x = a  # = b for right rectangle, (a+b)=/2

    for i in range(div):
        if method == 'left':
            s += f(x) * h
            x += h
        elif method == 'right':
            x += h
            s += f(x) * h
        else:
            s += h * f(x)
            x += h / 2
    return s


def trapeze_method(f, a, b, div=5, h=None):

    if h is None:
        h = (b - a) / div

    s = (f(a) + f(b)) / 2

    x = np.arange(a, b, h)

    for i in range(1, len(x)):
        s += f(x[i])

    return s * h


def simpson_method(f, a, b, h=None):

    def even(x):
        return x % 2 == 0

    if h is None:
        h = (b - a) / 2

    x_n = np.arange(a, b, h)
    x_odd = [x_n[i] for i in np.arange(len(x_n)) if even(i) == 0]
    x_even = [x_n[i] for i in np.arange(len(x_n)) if even(i)]

    s = 0

    for i in np.arange(len(x_odd)):
        s += 2 * f(x_even[i]) + 4 * f(x_odd[i])

    return s* h/3


def polynomial(c_list):

    def func(x):
        sum = 0
        for i in range(len(c_list)):
            sum += c_list[i] * (x ** i)
        return sum

    return func


def main():
    c = [0.4, 0.3, 0.2, 0.1, 0.2]
    a = -2.5
    b = 3.1
    f = polynomial(c)
    h = 0.01

    rectangle = rectangle_method(f, a, b)
    trapeze = trapeze_method(f, a, b)
    simpson = simpson_method(f, a, b)
    print('Simple methods \nrectangle: {},'
          ' trapeze: {}, simpson: {}'.format(rectangle, trapeze, simpson))

    rectangle = rectangle_method(f, a, b, h=h)
    trapeze = trapeze_method(f, a, b, h=h)
    simpson = simpson_method(f, a, b, h=h)
    print('\nGeneralized methods with h = {} \n'
          ' rectangle: {}, trapeze: {}, simpson: {}'.format(h, rectangle, trapeze, simpson))


if __name__ == '__main__':
    main()