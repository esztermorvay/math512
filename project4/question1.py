import math
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def generate_brownian_motion(t, drift, variance, multiplier = 1, dt=1):
    # Generate a brownian motion process with time t, uses recursion
    if t <= 0:
        # base case
        return 0,[0]
    random_normal = np.random.normal(0, 1)
    prev, list_of_values = generate_brownian_motion(t - dt, drift, variance, dt=dt)
    current = prev + drift + variance * random_normal*multiplier
    list_of_values.append(current)
    return current, list_of_values


def generate_brownian_motion_loop(t, drift, variance, multiplier = 1, dt=1):
    list_of_values = [0]
    t_val = 0
    current = 0
    prev_t = 0
    if t == 0:
        return 0, [0]
    while t_val <= t:
        t_val += dt
        random_normal = np.random.normal(0, 1)
        current = current + drift*dt + variance * math.sqrt(dt)*random_normal*multiplier
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

def q1_a():
    sample_size = 10000
    t = 2
    drift = 0
    variance = 1

    paths = []
    results = pd.DataFrame(columns=['L', 'mean', 'variance'])
    results_antithetic = pd.DataFrame(columns=['L', 'mean', 'variance'])
    # part_a_results = pd.DataFrame(columns=['L', 'mean', 'variance'])
    # part_a_results_antithetic = pd.DataFrame(columns=['L', 'mean', 'variance'])
    # part_b_results = pd.DataFrame(columns=['L', 'mean', 'variance'])
    # part_b_results_antithetic = pd.DataFrame(columns=['L', 'mean', 'variance'])
    # for every value of L to test, we have 100,000 samples and take the mean at each timestep
    # for exp in range(0,15):
    #     L = 2**exp
    Ls = []
    times = []
    L = 1000
    a_means = []
    a_variances = []
    b_means = []
    b_variances = []
    c_means = []
    c_variances = []
    t_s = []
    for t_ in range(0,2100,100):
        t_s.append(t_/1000)
    # for L in range(10,1000,20):
    # for L in range(10000,10001):
    for t in t_s:
        print("t: ", t)
        Ls.append(L)
        dWs = []
        samples = np.zeros(sample_size)
        samples_a = np.zeros(sample_size)
        samples_b = np.zeros(sample_size)
        samples_c = np.zeros(sample_size)
        samples_positives = np.zeros(sample_size // 2)
        samples_negatives = np.zeros(sample_size // 2)
        # calculating the runtime for each L
        dt = t / L
        # generate ts list
        # s = 0
        # ts = [0]
        # while s <= t:
        #     ts.append(s)
        #     s += dt

        start_time = time.time()
        for i in range(sample_size):
            # print("L: ", L, " percent done: ", i*100/sample_size, "%")
            # generate a brownian motion process

            current, all = generate_brownian_motion_loop(t, drift, variance, dt=t/L)
            total_a = 0
            for j in range(0, len(all)-1):
                # total_a += abs(all[j+1] - all[j])
                # total_a += (all[j + 1] - all[j])**2
                total_a += (all[j + 1] - all[j]) * all[j]
                # total_a += ((all[j] + all[j+1])/2)*(all[j + 1] - all[j])
                # total_a += all[j]*all[j]*dt
            samples_a[i] = total_a
            total_b = 0
            for j in range(0, len(all)-1):
                # total_a += abs(all[j+1] - all[j])
                # total_a += (all[j + 1] - all[j])**2
                # total_a += (all[j + 1] - all[j]) * all[j]
                total_b += ((all[j] + all[j+1])/2 + np.random.normal(0, 1))*(all[j + 1] - all[j])
                # total_a += all[j]*dt
            samples_b[i] = total_b
            samples[i] = total_a
            samples[i] = total_a*total_a
            total_c = 0
            for j in range(0, len(all)-1):
                # total_a += abs(all[j+1] - all[j])
                # total_a += (all[j + 1] - all[j])**2
                # total_a += (all[j + 1] - all[j]) * all[j]
                # total_b += ((all[j] + all[j+1])/2)*(all[j + 1] - all[j])
                # total_a += all[j]*dt
                total_c += dt
            samples_c[i] = total_c

            # using the trapezoidal method
            samples[i] = np.trapz(all,dx=dt)
            # if i < 50:
            #     plt.plot(ts, all)
            if i < sample_size//2:
                samples_positives[i] = samples[i]
            # else:
            #     break
            if len(paths) < 100:
                paths.append(all)
        end_time = time.time()
        for i in range(sample_size//2):
            current, all = generate_brownian_motion_loop(t, drift, variance,multiplier=-1, dt=t/L)
            total_a = 0
            for j in range(0, len(all) - 1):
                # total_a += abs(all[j + 1] - all[j])
                # total_a += (all[j + 1] - all[j]) ** 2
                # total_a += (all[j + 1] - all[j]) * all[j]
                # total_a += ((all[j] + all[j+1])/2)*(all[j + 1] - all[j])
                total_a += all[j] * all[j] * dt
            samples_negatives[i] = np.trapz(all,dx=dt)
            # samples_negatives[i] = total_a*total_a

        time_taken = (end_time - start_time)/sample_size
        times.append(time_taken)
        mean_norm = np.mean(samples)
        print(np.median(samples))
        variance_norm = np.var(samples)
        # add to the dataframe
        results = results.append({'L': L, 'mean': mean_norm, 'variance': variance_norm}, ignore_index=True)
        mean_pos = np.mean(samples_positives)
        mean_neg = np.mean(samples_negatives)
        mean = 0.5 * (mean_pos + mean_neg)
        cov = np.cov(samples_positives, samples_negatives)
        variance_ = 0.25 * (np.var(samples_positives) + np.var(samples_negatives) + 2 * cov[0][1])
        results_antithetic = results_antithetic.append({'L': L, 'mean': mean, 'variance': variance_}, ignore_index=True)

        a_means.append(np.mean(samples_a))
        a_variances.append(np.var(samples_a))
        b_means.append(np.mean(samples_b))
        b_variances.append(np.var(samples_b))
        c_means.append(np.mean(samples_c))
        c_variances.append(np.var(samples_c))
        # getting the differences between samples[i+1] and samples[i]

    plt.title("Curves to get area under")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.savefig("curves.png")
    plt.show()
    print(results)
    print(results_antithetic)
    results.to_csv('results.csv')
    results_antithetic.to_csv('results_antithetic.csv')
    results = results.dropna()
    results_antithetic = results_antithetic.dropna()
    # plot the mean and the variance for both results and antithetic in one figure two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(results['L'], results['mean'], label='mean')
    ax1.plot(results_antithetic['L'], results_antithetic['mean'], label='antithetic mean')
    ax1.set_title('Mean')
    ax1.set_xlabel('L')
    ax1.set_ylabel('Mean')
    ax1.legend()
    ax2.plot(results['L'], results['variance'], label='variance')
    ax2.plot(results_antithetic['L'], results_antithetic['variance'], label='antithetic variance')
    ax2.set_title('Variance')
    ax2.set_xlabel('L')
    ax2.set_ylabel('Variance')
    ax2.legend()
    plt.tight_layout()
    plt.savefig("q1a.png")
    plt.show()
    tot_resuls = pd.DataFrame(results)
    tot_resuls['antithetic mean'] = results_antithetic['mean']
    tot_resuls['antithetic variance'] = results_antithetic['variance']
    tot_resuls.to_csv('tot_results.csv')

    # plot the time taken
    # plt.plot(Ls, times)
    # plt.title('Time taken to calculate each L')
    # plt.xlabel('L')
    # plt.ylabel('Time taken')
    # plt.savefig('time_taken.png')
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_s, a_means, label='mean of A')
    ax1.plot(t_s, b_means, label='mean of B')
    ax1.plot(t_s, c_means, label='mean of C')
    ax1.set_title('Mean')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Mean')
    ax1.legend()
    ax2.plot(a_variances, label='variance of A')
    ax2.plot(b_variances, label='variance of B')
    ax2.plot(c_variances, label='variance of C')
    ax2.set_title('Variance')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Variance')
    ax2.legend()
    plt.tight_layout()
    plt.savefig("q2.png")
    plt.show()



def main():
    q1_a()

main()
