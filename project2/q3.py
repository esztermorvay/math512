import pickle
import random
import time

import numpy as np
# from matplotlib import pyplot as plt
from ncephes import bdtri

partASum = 0
partBSum = 0
def bernoulli(n, p):
    # simulate n trials of a bernoulli trial with probability p and adds the sum
    return np.random.binomial(n, p)

def partB():
    # time how long this takes
    # get the current time
    start = time.time()
    results = []
    results2 = []
    for i in range(0, 6000):
        successes = 0
        for i in range(100):
            num = random.uniform(0,1)
            if num <= .7:
                successes += 1
        # use the inverse binomial cdf from cprob
        result = bdtri(successes, 100, .7)
        results.append(result)
    end = time.time()
    global partBSum
    partBSum += end - start
    # print("Time to run part B: " + str(end - start))
    # print(results)

    # pickle.dump(results, open("results.p", "wb"))
    #unpickle
    results = pickle.load(open("results.p", "rb"))
    # multiply everythin in results by 100
    for i in range(len(results)):
        results[i] *= 100
    # plt.hist(results, bins=17)
    # plt.title("Binomial Distribution Using Inverse Binomial CDF")
    # plt.xlabel("Successes")
    # plt.ylabel("Frequency")
    # plt.savefig("q3b.png")

def partA():
    # 6000 trials with prob .7
    global partASum

    start = time.time()
    results = []
    for i in range(0, 6000):
        results.append(bernoulli(100, .7))
    end = time.time()
    partASum += end - start
    # print("Time to run part A: " + str(end - start))
    # plot a histogram of results
    # plt.hist(results, bins=15)
    # plt.title("Binomial Distribution Using Bernoulli Trials")
    # plt.xlabel("Sum of Successes")
    # plt.ylabel("Frequency")
    # plt.savefig("q3.png")
    # plt.show()
    # print(results)
    # calculate the percent of trials that had less than or equal to 70 successes
    count = 0
    for i in results:
        if i <= 70:
            count += 1
    # print("Proportion of trials with less than or equal to 70 successes: ", count / 6000)

def main():
    for i in range(1000):
        partA()
        partB()
        if i % 100 == 0:
            print(i)
    print("Average time for part A: ", partASum / 1000)
    print("Average time for part B: ", partBSum / 1000)
    # partA()
    # partB()
if __name__ == "__main__":
    main()


