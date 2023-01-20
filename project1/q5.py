


"""
# Write a routine (call it) bisect to find the root of the function 𝑓(𝑥) = √𝑥 − 1.1 starting from the interval [0,2], with 𝑎𝑡𝑜𝑙 = 1. 𝑒 − 8.
# The routine should return the root and the number of iterations it took to find the root.
"""
import math
import matplotlib.pyplot as plt

values = []
roots = []
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
    It also keeps track of each range and root estimate in case we need it for plotting
    """

    c = (a + b) / 2
    values.append(abs(b-a))
    roots.append(c)
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


def f(x):
    return math.sqrt(x) - 1.1

def run_routine():
    root, iterations = bisect(f, 0, 2, 1e-8)
    print("Root estimate: ", root)
    print("Iterations it took: ", iterations)

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

def main():
    run_routine()
    plot_values()
    plot_roots()

if __name__ == "__main__":
    main()