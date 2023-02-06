import numpy as np
from matplotlib import pyplot as plt


def bernoulli(n, p):
    # simulate n trials of a bernoulli trial with probability p and adds the sum
    return np.random.binomial(n, p)


def main():
    # 6000 trials with prob .7
    results = []
    for i in range(0, 6000):
        results.append(bernoulli(100, .7))
    # plot a histogram of results
    plt.hist(results, bins=15)
    plt.title("Binomial Distribution Using Bernoulli Trials")
    plt.xlabel("Sum of Successes")
    plt.ylabel("Frequency")
    plt.savefig("q3.png")
    plt.show()
    # print(results)
    # calculate the percent of trials that had less than or equal to 70 successes
    count = 0
    for i in results:
        if i <= 70:
            count += 1
    print("Proportion of trials with less than or equal to 70 successes: ", count / 6000)

if __name__ == "__main__":
    main()


