


"""
# Write a routine (call it) bisect to find the root of the function 𝑓(𝑥) = √𝑥 − 1.1 starting from the interval [0,2], with 𝑎𝑡𝑜𝑙 = 1. 𝑒 − 8.
# The routine should return the root and the number of iterations it took to find the root.
"""
import math


def bisect(f, a, b, atol):
    # f is the function
    # a is the lower bound
    # b is the upper bound
    # atol is the absolute tolerance
    # return the root and the number of iterations it took to find the root
    if abs(b-a) < atol:
        return a, 0
    else:
        c = (a+b)/2
        if f(a)*f(c) < 0:
            root, iterations = bisect(f, a, c, atol)
        else:
            root, iterations = bisect(f, c, b, atol)
        iterations += 1
        return root, iterations


def f(x):
    return math.sqrt(x) - 1.1

def run_routine():
    root, iterations = bisect(f, 0, 2, 1e-8)
    print("Root: ", root)
    print("Iterations: ", iterations)
def main():
    run_routine()

if __name__ == "__main__":
    main()