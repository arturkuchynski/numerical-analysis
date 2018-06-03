import numpy as np
from task_10 import trapeze_method, simpson_method


def chebyshev_method(f, a, b, c, n):
    s = 0

    for c in c[n]:
        x = (b + a) / 2 + ((b - a) / 2) * c
        s += f(x)

    s *= (b - a) / n
    return s


def f(x):
    return np.e ** (-2 * np.sin(x))


def main():
    a, b = -1.5, 0

    c = {3: [0.707107, 0, -0.707107],
         4: [0.794654, 0.187592, -0.187592, -0.794654],
         }

    cheb_3 = chebyshev_method(f, a, b, c, n=3)
    cheb_4 = chebyshev_method(f, a, b, c, n=4)
    print('Chebyshev method\n with n = 3:'
          ' {}\nwith n = 4: {}'.format(cheb_3, cheb_4))

    print('Generalized methods')
    simps = simpson_method(f, a, b, h=0.01)
    trapez = trapeze_method(f, a, b, h=0.01)
    print('trapeze: {}, simpson: {}'.format(trapez, simps))


if __name__ == '__main__':
    main()
