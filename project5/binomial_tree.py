from datetime import datetime
import math
import pandas as pd
from matplotlib import pyplot as plt


def get_historical_volatility():
    df = pd.read_csv("TSLA.csv")
    df["Close_shifted"] = df["Close"].shift(1)
    df = df.dropna()
    df["log_ratio"] = df.apply(lambda row: math.log(row["Close"] / row["Close_shifted"]), axis=1)
    print(df["log_ratio"])
    # mean = df["Close"].mean()
    mean = df["log_ratio"].mean()
    # std = df["log_ratio"].std()
    n = len(df)
    sum = 0
    for index, row in df.iterrows():
        sum += (row["log_ratio"] - mean) ** 2
    std = math.sqrt(sum / (n - 1))
    std = std * math.sqrt(252)
    price = df["Close"].iloc[-1]
    return std, price

def get_delta_t(frac_ = 1):
    # this represents the depth of the binomial tree
    start = datetime(2020, 4, 20)
    end = datetime(2020, 6, 16)
    delta = end - start
    return  (delta.days / frac_)* (1/365)

def get_u(sigma, dt):
    return math.e ** (sigma * math.sqrt(dt))

def get_d(sigma, dt):
    return math.e ** (-sigma * math.sqrt(dt))

def get_fu(S_u, strike_price, call=True):
    if call:
        return max(0, S_u - strike_price)
    else:
        return max(0, strike_price - S_u)

def get_fd(S_d, strike_price, call=True):
    if call:
        return max(0, S_d - strike_price)
    else:
        return max(0, strike_price - S_d)

def get_q(r, dt, u, d, call=True):
    return (math.e ** (r * dt) - d) / (u - d)
    if call:
        return (math.e ** (r * dt) - d) / (u - d)
    else:
        return (u - math.e ** (r * dt)) / (u - d)

def get_f(f_u, f_d, q, r, dt):
    return math.e ** (-r * dt) * (q * f_u + (1 - q) * f_d)


def get_delta(f_u, f_d, S_u, S_d, call=True):
    if call:
        return (f_u - f_d) / (S_u - S_d)
    else:
        return (f_d - f_u) / (S_d - S_u)

def get_last_iter(S, dt, sigma, strike_price, r, call=True):
    u = get_u(sigma, dt)
    d = get_d(sigma, dt)
    S_u = S * u
    S_d = S * d
    f_u = get_fu(S_u, strike_price, call)
    f_d = get_fd(S_d, strike_price, call)
    q = get_q(r, dt, u, d, call)
    f = get_f(f_u, f_d, q, r, dt)
    # delta = get_delta(f_u, f_d, S_u, S_d, call)
    return f

def populate_tree(S, u, d, num_levels):
    tree = [[S]]
    for i in range(1,num_levels):
        level = []
        for j in range(0, len(tree[i-1])):
            level.append(tree[i-1][j] * u)
            level.append(tree[i-1][j] * d)
        tree.append(level)
    return tree

def print_tree(tree):
    for level in tree:
        print(level)

def q_b(S, frac, sigma, strike_price, r):
    # iterate through the binomial tree
    dt = get_delta_t(frac)
    # dt = 0.25
    t = frac
    u = get_u(sigma, dt)
    d = get_d(sigma, dt)
    # u = 1.1
    # d = 0.9
    q = get_q(r, dt, u, d)
    tree = populate_tree(S, u, d, frac+1)
    # print_tree(tree)
    # getting the last iteration first
    f_values = []
    for i in range(0, len(tree[-1]), 2):
        S_u = tree[-1][i]
        S_d = tree[-1][i+1]
        f_u = get_fu(S_u, strike_price)
        f_d = get_fd(S_d, strike_price)
        f = get_f(f_u, f_d, q, r, dt)
        f_values.append(f)

    # now we iterate backwards
    while len(f_values) > 1:
        new_f_values = []
        for i in range(0, len(f_values), 2):
            print(f_values)
            f_u = f_values[i]
            f_d = f_values[i+1]
            f = get_f(f_u, f_d, q, r, dt)
            new_f_values.append(f)
        f_values = new_f_values

    print(f_values[0])
    return f_values[0]

def q_put(S, frac, sigma, strike_price, r):
    # european put option
    # iterate through the binomial tree
    dt = get_delta_t(frac)
    # dt = 0.25
    t = frac
    u = 1.2
    d = 0.8
    dt = 1
    q = get_q(r, dt, u, d, False)
    tree = populate_tree(S, u, d, frac + 1)
    print_tree(tree)
    # getting the last iteration first
    f_values = []
    for i in range(0, len(tree[-1]), 2):
        S_u = tree[-1][i]
        S_d = tree[-1][i + 1]
        f_u = get_fu(S_u, strike_price, False)
        f_d = get_fd(S_d, strike_price, False)
        f = get_f(f_u, f_d, q, r, dt)
        f_values.append(f)

    # now we iterate backwards
    while len(f_values) > 1:
        print(f_values)
        new_f_values = []
        for i in range(0, len(f_values), 2):
            f_u = f_values[i]
            f_d = f_values[i + 1]
            f = get_f(f_u, f_d, q, r, dt)
            new_f_values.append(f)
        f_values = new_f_values

    print(f_values[0])

def q_d(S, frac, sigma, strike_price, r):
    # now calculating a put American option
    dt = get_delta_t(frac)
    u = get_u(sigma, dt)
    d = get_d(sigma, dt)
    # u = 1.2
    # d = 0.8
    # dt = 1
    q = get_q(r, dt, u, d, False)
    tree = populate_tree(S, u, d, frac+1)
    print_tree(tree)
    # since its an american option we now have to account for settlement price at each step
    # getting the last iteration first
    f_values = []
    for i in range(0, len(tree[-1]), 2):
        S_u = tree[-1][i]
        S_d = tree[-1][i+1]
        f_u = get_fu(S_u, strike_price, False)
        f_d = get_fd(S_d, strike_price, False)
        f1 = get_f(f_u, f_d, q, r, dt)
        s = tree[-2][i // 2]
        f2 = strike_price - s
        f_values.append(max(f1, f2))


    counter = len(tree)- 3
    while len(f_values) > 1:
        print(f_values)

        new_f_values = []
        for i in range(0, len(f_values), 2):
            f_u = f_values[i]
            f_d = f_values[i+1]
            f1 = get_f(f_u, f_d, q, r, dt)
            # S_u = tree[counter][i]
            # S_d = tree[counter][i + 1]
            # f2 = get_f(f_u, f_d, q, r, dt)
            s = tree[counter][i//2]
            f2 = strike_price - s
            new_f_values.append(max(f1, f2))
        f_values = new_f_values

    print(f_values[0])



def main():
    sigma, price = get_historical_volatility()
    # price = 20
    r = 0.05
    print(sigma)
    print(price)

    strike_price = price+200
    # r = .12
    # strike_price = 21
    # q_b(price, 20, sigma, strike_price, r)
    # return 0
    # quesiton c
    option_vals = []
    depths = []
    ratios = []
    epsilon = .0000001
    for depth in range(1, 25):
        option_val = q_b(price, depth, sigma, strike_price, r)
        option_vals.append(option_val)
        depths.append(depth)
        if depth > 1:
            if option_vals[depth-2] == 0:
                ratios.append(0)
            else:
                ratios.append(option_val/(option_vals[depth-2]))
    plt.plot(depths, option_vals)
    plt.title("Option Value vs Time Partitions")
    plt.xlabel("Time Partitions")
    plt.ylabel("Option Value")
    plt.savefig("option_val.png")
    plt.ylim(0,.5)
    plt.show()
    # plot convergence rate
    plt.plot(depths[1:], ratios)
    plt.title("Convergence Rate vs Time Partitions")
    plt.xlabel("Time Partitions")
    plt.ylabel("Convergence Rate")
    plt.ylim(0,10)
    plt.savefig("convergence_rate.png")
    plt.show()
    print(ratios[-1])





    # an ex from the textbook
    # price = 50
    # strike_price = 52
    # r = 0.05
    # # q_put(price, 2, sigma, strike_price, r)
    # q_d(price, 2, sigma, strike_price, r)

    #question d
    # strike_price = price + 200
    # q_d(price, 20, sigma, strike_price, r)






main()
