

import os
import math
import plots
import numpy as np
import numpy.linalg
from first_system import FirstSystem
from second_system import SecondSystem

"""
    Laboratory task №4-5
    Variant 12, Artur Kuchynski
"""


def newton_method(system, init_x, init_y,  eps=1e-5):

    x_i, y_i = init_x, init_y

    while math.fabs(system.f1(x_i, y_i)) > eps and math.fabs(system.f2(x_i, y_i)) > eps:

        # Kramer method
        a1 = np.array([[system.f1(x_i, y_i), system.df1dy(x_i, y_i)],
                      [system.f2(x_i, y_i), system.df2dy(x_i, y_i)]])

        a2 = np.array([[system.df1dx(x_i, y_i), system.f1(x_i, y_i)],
                      [system.df2dx(x_i, y_i), system.f2(x_i, y_i)]])
        
        # jacobian matrix
        jacobian = np.array([[system.df1dx(x_i, y_i), system.df2dx(x_i, y_i)],
                            [system.df1dy(x_i, y_i), system.df2dy(x_i, y_i)]])

        if numpy.linalg.det(jacobian) == 0:
            return None, None
        else:
            x_i = x_i - numpy.linalg.det(a1) / numpy.linalg.det(jacobian)
            y_i = y_i - numpy.linalg.det(a2) / numpy.linalg.det(jacobian)

    return x_i, y_i


def iteration_method(system, init_x, init_y, eps=1e-5):

    derivative_matrix = np.array([[system.df1dy(init_x, init_y), system.df2dy(init_x, init_y)],
                           [system.df2dy(init_x, init_y), system.df2dx(init_x, init_y)]])

    # column for augmented matrix x
    x_matrix_column = np.array([-1, 0])
    lambda_11, lambda_12 = np.linalg.solve(derivative_matrix, x_matrix_column)
    # column for augmented matrix y
    y_matrix_column = np.array([0, -1])
    lambda_21, lambda_22 = np.linalg.solve(derivative_matrix, y_matrix_column)
    """
    def iterative_function_x(x, y):
        return x + lambda_11 * system.f1(x, y) + lambda_12 * system.f2(x, y)

    def iterative_function_y(x, y):
        return y + lambda_21 * system.f1(x, y) + lambda_22 * system.f2(x, y)
    """
    iterative_function_x = lambda x, y: x + lambda_11 * system.f1(x, y) + lambda_12 * system.f2(x, y)
    iterative_function_y = lambda x, y: y + lambda_21 * system.f1(x, y) + lambda_22 * system.f2(x, y)

    x_i, y_i = init_x, init_y

    # iterative step
    while math.fabs(system.f1(x_i, y_i)) > eps and math.fabs(system.f2(x_i, y_i)) > eps:
        x_temp, y_temp = x_i, y_i
        x_i = iterative_function_x(x_temp, y_temp)
        y_i = iterative_function_y(x_temp, y_temp)

    return x_i, y_i


def main():

    plots.display_plot()

    with np.errstate(all='ignore'):
        system_dialog(FirstSystem(), "first system")
        system_dialog(SecondSystem(), "1st root of second system")
        system_dialog(SecondSystem(), "2nd root of second system")


# clear screen of command prompt/terminal using the following method
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def system_dialog(system, system_number):

    print("\nSet approximated values for the {}".format(system_number))

    x = float(input("set value of x near the root: "))
    y = float(input("set value of y near the root: "))

    # clear_screen()

    newton_method_res = newton_method(system, x, y)
    iteration_method_res = iteration_method(system, x, y)
    print("Newton's method result: {} , {}".format(*newton_method_res))
    print("Iteration method result: {} , {}".format(*iteration_method_res))
    print("#######################################################################")


if __name__ == '__main__':
    main()
