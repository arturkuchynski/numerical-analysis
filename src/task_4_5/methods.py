import os
import math
import plots
import numpy as np
import numpy.linalg
from first_system import FirstSystem
from second_system import SecondSystem


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

    system_matrix = np.array([[system.df1dy(init_x, init_y), system.df2dy(init_x, init_y)],
                           [system.df2dy(init_x, init_y), system.df2dx(init_x, init_y)]])

    # making array of values for the variable x
    x_values_row = np.array([-1, 0])
    c11, c12 = np.linalg.solve(system_matrix, x_values_row) 
    # making array of values for the variable y
    y_values_row = np.array([0, -1])
    c21, c22 = np.linalg.solve(system_matrix, y_values_row)

    def iterative_function_x(x, y):
        return x + c11 * system.f1(x, y) + c12 * system.f2(x, y)

    def iterative_function_y(x, y):
        return y + c21 * system.f1(x, y) + c22 * system.f2(x, y)

    x_i, y_i = init_x, init_y
    # iterative step
    while math.fabs(system.f1(x_i, y_i)) > eps and math.fabs(system.f2(x_i, y_i)) > eps:
        x_temp, y_temp = x_i, y_i
        x_i = iterative_function_x(x_temp, y_temp)
        y_i = iterative_function_y(x_temp, y_temp)

    return x_i, y_i


def main():

    plots.display_plots()

    with np.errstate(all='ignore'):
        system_dialog(FirstSystem(), "first system")
        system_dialog(SecondSystem(), "1st point of second system,")
        system_dialog(SecondSystem(), "2nd point")


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
