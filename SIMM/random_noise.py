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

if __name__ == '__main__':
    n = 6
    Y = (np.random.uniform(low=0.0, high=1.0, size=(n, n)) > 0.5).astype(int)
    print(Y)
    Y_p = random_noise(Y, 3)
    print(Y_p)






