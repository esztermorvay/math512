import numpy as np
from matplotlib import pyplot as plt


def generate_brownian_motion(t, drift, variance, multiplier = 1):
    # Generate a brownian motion process with time t, uses recursion
    if t == 0:
        # base case
        return 0,[0]
    random_normal = np.random.normal(0, 1)
    prev, list_of_values = generate_brownian_motion(t - 1, drift, variance)
    current = prev + drift + variance * random_normal*multiplier
    list_of_values.append(current)
    return current, list_of_values


def get_variance(values):
    mean = np.mean(values)
    variance = 0
    for value in values:
        variance += (value - mean) ** 2
    return variance / len(values)

def get_mean(valus):
    return np.mean(valus)

def question1():
    # Estimating the expected value of E[W(3)^2 + cosW(3)]
    # with drift parameter 0 and variance parameter 1

    # figure to plot the paths
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # a 10,000 size numpy array to hold all the samples:
    samples = np.zeros(1000000)
    samples_positives = np.zeros(500000)
    samples_negatives = np.zeros(500000)
    paths = []
    for i in range(1000000):
        # generate a brownian motion process
        current, all = generate_brownian_motion(3, 0, 1)
        samples[i] = current**2 + np.cos(all[-1])
        if i < 500000:
            samples_positives[i] = samples[i]
        else:
            break
        if len(paths) < 100:
            paths.append(all)
    for i in range(500000):
        current, all = generate_brownian_motion(3, 0, 1, -1)
        samples_negatives[i] = current**2 + np.cos(all[-1])

    # get the mean of samples negative and samples positive
    mean_pos = np.mean(samples_positives)
    mean_neg = np.mean(samples_negatives)
    mean = 0.5*(mean_pos + mean_neg)
    print("mean : ", mean)
    # calculating the variance
    # get the covariance of the two samples
    cov = np.cov(samples_positives, samples_negatives)
    variance = 0.25*(np.var(samples_positives) + np.var(samples_negatives) + 2*cov[0][1])
    print("variance : ", variance)
    # plot the paths
    # for i in range(100):
    #     ax.plot(paths[i])
    # plt.title("Paths of Brownian Motion")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.savefig("q1.png")
    #
    # # plot paths with all the values transformed
    # plt.clf()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(len(paths)):
    #     path = paths[i]
    #     for j in range(len(path)):
    #         path[j] = path[j]**2 + np.cos(path[j])
    #     ax.plot(path)
    # plt.title("Paths of Brownian Motion Expected Value")
    # plt.xlabel("Time")
    # plt.ylabel("Expected Value")
    #
    # plt.savefig("q1b.png")

    # print("The expected value of E[W(3)^2 + cosW(3)] is: ", np.mean(samples))
    # print("The variance of E[W(3)^2 + cosW(3)] is: ", np.var(samples))

def calculate_geometric_brownian_motion(S_0, r, sigma, W_t, t):
    return S_0 * np.exp(sigma * W_t + (r - sigma**2/2)*t )
def question2():
    r = 0.07
    sigma = 0.20
    S_0 = 80
    samples = np.zeros(1000000)
    samples_positives = np.zeros(500000)
    samples_negatives = np.zeros(500000)
    paths = []
    for i in range(1000000):
        current, all = generate_brownian_motion(2, 0, 1)
        samples[i] = calculate_geometric_brownian_motion(S_0, r, sigma, current, 2)
        if i < 500000:
            samples_positives[i] = samples[i]
        else:
            break
        if len(paths) < 100:
            paths.append(all)
    for i in range(500000):
        current, all =  generate_brownian_motion(2, 0, 1, -1)
        samples_negatives[i] = calculate_geometric_brownian_motion(S_0, r, sigma, current, 2)
    mean_pos = np.mean(samples_positives)
    mean_neg = np.mean(samples_negatives)
    mean = 0.5 * (mean_pos + mean_neg)
    print("mean : ", mean)
    # calculating the variance
    # get the covariance of the two samples
    cov = np.cov(samples_positives, samples_negatives)
    variance = 0.25 * (np.var(samples_positives) + np.var(samples_negatives) + 2 * cov[0][1])
    print("variance : ", variance)
    return
    # plot the paths
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(50):
        path = paths[i]
        for j in range(len(path)):
            # transform the paths
            path[j] = calculate_geometric_brownian_motion(S_0, r, sigma, path[j], j)
        ax.plot(path)
    plt.title("Paths of Geometric Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig("q2.png")
    print("The expected value of E[S(2)] is: ", np.mean(samples))
    print("The variance of E[S(2)] is: ", np.var(samples))





def main():
    # question1()
    question2()
if __name__ == "__main__":
    main()