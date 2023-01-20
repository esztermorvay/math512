"""
This file is for q5 and q6 code
"""


import math
import matplotlib.pyplot as plt
import numpy as np

# these are some lists that will be used to plot the values
# we update the values as each function is called
values = []
roots = []
errors = []
convergence = []

values_newton = []
roots_newton = []
x_vals_newton = []
errors_newton = []
convergence_newton = []

values_fixed_point = []
roots_fixed_point = []
x_vals_fixed_point = []
errors_fixed_point = []
convergence_fixed_point = []

"""
Function definitions for funcs we are gonna use
"""
def f(x):
    return np.sqrt(x) - 1.1

def df(x):
    # the derivative of the function
    return 1/(2*np.sqrt(x))

def g(x):
    return x-f(x)

def get_absolute_error(root):
    return abs(1.21 - root)

"""
SECTION FOR METHODS CODE
"""
def bisect(f, a, b, atol):
    """
    :param f: function
    :type f: function
    :param a: lower bound
    :type a: float
    :param b: upper bound
    :type b: float
    :param atol: tolerance
    :type atol: float
    :return: root, # iterations
    :rtype: tuple

    This function recursively finds the root of a function f(x) using the bisection method.
    It also keeps track of each range interval, root estimate, and error at each iteration so we can plot it
    """

    c = (a + b) / 2
    values.append(abs(b-a))
    roots.append(c)
    errors.append(get_absolute_error(c))
    if abs(b-a) < atol:
        return c, 0
    else:
        # set c to be the lower or upper bound
        if f(a)*f(c) < 0:
            root, iterations = bisect(f, a, c, atol)
        else:
            root, iterations = bisect(f, c, b, atol)
        # increase the number of iterations each tim e a recursive call was made
        iterations += 1
        return root, iterations


def newton(f, df, start, atol):
    """
    :param f: function
    :type f: function
    :param df: derivative of the function
    :type df: function
    :param start: initial guess
    :type start: float
    :param atol: tolerance
    :type atol: float
    :return: root, # iterations
    :rtype: tuple

    This function recursively finds the root of a function f(x) using Newton;s method.
    It also keeps track of each root estimate, and error at each iteration so we can plot it
    """
    current = start - f(start) / df(start)
    roots_newton.append(current)
    x_vals_newton.append(start)
    errors_newton.append(get_absolute_error(current))
    if abs(current - start) < atol:
        return current, 0
    else:
        root, iterations = newton(f, df, current, atol)
        # again, increase iterations when recursive call is made
        iterations += 1
        return root, iterations

def fixed_point_loop(g, start, atol):
    """
    :param g: function
    :type g: function
    :param start: initial guess
    :type start: float
    :param atol: tolerance
    :type atol: float
    :return: root, # iterations
    :rtype: tuple

    This function finds the root of f(x) using the fixed point method.
    It also keeps track of each root estimate, and error at each iteration so we can plot it.
    I implemented this as a loop instead of recursively so there would be some variety.
    """
    current = start
    roots_fixed_point.append(current)

    iterations = 0
    done = False
    while not done:
        current = g(current)
        roots_fixed_point.append(current)
        errors_fixed_point.append(get_absolute_error(current))
        if abs(current - start) < atol:
            done = True
            return current, iterations
        else:
            iterations += 1
            start = current



"""
This is the section for routines for each method
"""

def run_routine():
    print("Bisection Method")
    root, iterations = bisect(f, 0, 2, 1e-8)
    print("Root estimate: ", root)
    print("Iterations it took: ", iterations)
    print("Absolute error: ", get_absolute_error(root))
    print()

def run_routine_newton():
    print("Newton's Method")
    root, iterations = newton(f, df, 1e-8, 1e-8)
    print("Root estimate: ", root)
    print("Iterations it took: ", iterations)
    print("Absolute error: ", get_absolute_error(root))
    print()

def run_fixed_point_routine():
    print("Fixed Point Method")
    root, iterations = fixed_point_loop(g, 1e-8, 1e-8)
    print("Root estimate: ", root)
    print("Iterations it took: ", iterations)
    print("Absolute error: ", get_absolute_error(root))
    print()


"""
This section is for plotting functions
"""

def plot_values():
    plt.plot(values)
    plt.title("Interval Size vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Interval Size")
    plt.savefig("q5.png")
    plt.show()

def plot_roots():
    plt.plot(roots)
    plt.title("Root estimate vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Root estimate")
    plt.savefig("q5.2.png")
    plt.show()

def plot_errors():
    plt.plot(errors)
    plt.title("Absolute Error vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.savefig("q5.3.png")
    plt.show()

def plot_all_errors(use_cutoffs=False):
    if use_cutoffs:
        plt.plot(errors, label="Bisection")
        plt.plot(range(3,len(errors_newton)), errors_newton[3:], label="Newton")
        plt.plot(range(1,len(errors_fixed_point)), errors_fixed_point[1:], label="Fixed Point")
    else:
        plt.plot(errors, label="Bisection")
        plt.plot(errors_newton, label="Newton")
        plt.plot(errors_fixed_point, label="Fixed Point")
    plt.legend()
    plt.title("Absolute Error vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.savefig("q5."+ str(use_cutoffs)+"allerrors.png")
    plt.show()

def plot_function(f, lower_bound, upper_bound):
    x = np.linspace(lower_bound, upper_bound, 100)
    y = f(x)
    plt.plot(x, y)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()

def plot_function_and_roots(f, lower_bound, upper_bound, method=None):
    """
    :param f:
    :param lower_bound:
    :param upper_bound:
    :return:
    plots the function and the estimates of the roots using the 3 methods found

    """

    x = np.linspace(lower_bound, upper_bound, 100)
    y = f(x)
    plt.plot(x, y, label="Function")
    plt.axhline(y=0, color='r', linestyle='-')

    if method is None:
        plt.plot(roots, f(roots), '.', label="Bisection", alpha=0.5)
        plt.plot(roots_newton, f(roots_newton), '.',  label="Newton", alpha=0.5)
        plt.plot(roots_fixed_point, f(roots_fixed_point), '.', label="Fixed Point", alpha=0.5, color='purple')
    elif method == 'bisection':
        plt.plot(roots, f(roots), 'o', label="Bisection")
    elif method == 'newton':
        plt.plot(roots_newton, f(roots_newton), 'o', label="Newton", color='green')
    elif method == 'fixed_point':
        plt.plot(roots_fixed_point, f(roots_fixed_point), 'o', label="Fixed Point", color='purple')
    plt.legend()
    plt.title("Root estimates vs. True Function")
    save_name = "q5.method" + str(method) + ".png"
    plt.savefig(save_name)
    plt.show()


def plot_convergence_rate():
    """
    This is a function to plot the convergence rates
    :return:
    """
    for i in range(1, len(roots)):
        convergence.append((roots[i]) / (roots[i - 1]))
    for i in range(1, len(roots_newton)):
        convergence_newton.append((roots_newton[i]) / (roots_newton[i - 1]))
    for i in range(1, len(roots_fixed_point)):
        convergence_fixed_point.append((roots_fixed_point[i]) / (roots_fixed_point[i - 1]))
    print("Bisection Method Convergence Rate: ", convergence[-1])
    print("Newton's Method Convergence Rate: ", convergence_newton[-1])
    print("Fixed Point Method Convergence Rate: ", convergence_fixed_point[-1])
    plt.plot(convergence, label="Bisection")
    plt.plot(range(2, len(convergence_newton)), convergence_newton[2:], label="Newton")
    plt.plot(range(1, len(convergence_fixed_point)), convergence_fixed_point[1:], label="Fixed Point")
    plt.legend()
    plt.title("Convergence Rate vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Convergence Rate")
    plt.savefig("q5.4.png")
    plt.show()

"""
This section is for running the routines
"""
def main():
    run_routine()
    plot_values()
    plot_roots()
    plot_errors()
    plot_function(f, -10, 10)
    run_routine_newton()
    run_fixed_point_routine()
    plot_function_and_roots(f, -10, 2, method='bisection')
    plot_function_and_roots(f, -10, 2, method='newton')
    plot_function_and_roots(f, -10, 2, method='fixed_point')
    plot_function_and_roots(f, -10, 2)
    plot_convergence_rate()
    plot_all_errors()
    plot_all_errors(use_cutoffs=True)

if __name__ == "__main__":
    main()