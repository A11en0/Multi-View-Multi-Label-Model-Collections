import numpy as np


def random_noise(Y, noise_num=3):
    N, M = Y.shape
    for i in range(N):
        neg_idx = np.where(Y[i] == 0)[0]
        if len(neg_idx) < noise_num:
            Y[i, neg_idx] = 1

        else:
            choose_idx = np.random.choice(neg_idx, noise_num, replace=False)
            Y[i, choose_idx] = 1

    return Y

def random_noise_p_r(Y, noise_rate, noise_num=3):
    N, M = Y.shape
    noise_nums = np.zeros((N, 1))
    noise_num_p = int(N * noise_rate)
    rand_idx_p = np.random.permutation(N)
    choose_idx_p = rand_idx_p[:noise_num_p]
    for p in range(noise_num_p):
        i = choose_idx_p[p]
        neg_idx = np.where(Y[i] == 0)[0]
        if len(neg_idx) < noise_num:
            Y[i, neg_idx] = 1
            noise_nums[i] = len(neg_idx)
        else:
            choose_idx = np.random.choice(neg_idx, noise_num, replace=False)
            Y[i, choose_idx] = 1
            noise_nums[i] = noise_num

    return Y, noise_nums

if __name__ == '__main__':
    n = 10
    Y = (np.random.uniform(low=0.0, high=1.0, size=(n, n)) > 0.5).astype(int)
    print(Y)
    print()
    # Y_p = random_noise(Y, 3)
    Y_p, noise_nums = random_noise_p_r(Y, 0, 3)
    print(Y_p)
    print(noise_nums)






