import matplotlib.pyplot as plt
import numpy as np


def display_plot():
    x_first_range = np.arange(-7, 7, 1e-4)
    x_sec_range = np.arange(-2, 2, 1e-4)

    with np.errstate(all='ignore'):
        y1_1 = [np.arccos(- x + 0.5) + 1 for x in x_first_range]
        y2_1 = [np.cos(x) + 3 for x in x_first_range]
        y1_2 = [np.arcsin(1.5*x - 0.1) - x for x in x_sec_range]
        y2_2 = [np.sqrt(1 - x**2) for x in x_sec_range]
        y2_2_negative = [- np.sqrt(1 - x**2) for x in x_sec_range]

    # first system
    # plt.axis([-3, 3, -4, 4])
    plt.subplot(211)
    plt.grid(True)
    plt.plot(x_first_range, y1_1, 'g')
    plt.plot(x_first_range, y2_1)
    plt.title('First system')
    # second system
    plt.subplot(212)
    plt.grid(True)
    plt.plot(x_sec_range, y1_2)
    plt.plot(x_sec_range, y2_2, 'y')
    plt.plot(x_sec_range, y2_2_negative, 'y')
    plt.title('Second system')
    plt.show()
