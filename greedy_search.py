
import pdb
import numpy as np


def greedy_oracle(scores, sim, k, sigma, lam_da):
    """
        sim = X.T * X
    """
    m_set = set(range(sim.shape[0]))
    i = np.argmax(scores)
    S = [i] # the first item
    C = 1.0 / (sim[i, i] + sigma**(2))

    for j in range(1, k):
        c_set = m_set - set(S)
        ii = np.array(list(c_set))
        jj = np.array(S)

        sigma_is = sim[ii, :][:, jj]
        if j > 1:
            tmp = np.sum(np.dot(sigma_is, C) * sigma_is, axis=1) + sigma**(2)
        else:
            tmp = C * np.sum(sigma_is * sigma_is, axis=1) + sigma**(2)

        tmp = scores[ii]  + 0.5 * lam_da * np.log(2 * np.pi * np.e * tmp)
        k = np.argmax(tmp)
        S.append(ii[k])

        kk = np.array(S)
        C = np.linalg.inv(sim[kk, :][:, kk] + sigma**(2) * np.identity(len(S)))

    return S
            

def oracle(args, scores, features):
    s = []
    sigma = args.sigma
    k = args.num_recommendation
    m = args.movie_dim
    c = np.power(sigma, -2)
    for j in range(k):
        delta_r = {}
        delta_g = {}
        for i in range(m):
            if i not in s:
                if s:
                    tmp = np.matmul(np.transpose(features[:, i]),
                                    features[:, s])
                    delta_r[i] = scores[i]
                    tmp_1 = np.matmul(tmp, c)
                    delta_g[i] = 1. / 2 * np.log(
                        2 * np.pi * np.e * (sigma ** 2 + np.matmul(tmp_1, np.transpose(tmp)))
                    )
                else:
                    # in case of empty s
                    tmp = 0
                    delta_r[i] = scores[i]
                    delta_g[i] = 1. / 2 * np.log(
                        2 * np.pi * np.e * (sigma ** 2)
                    )
        max_value = -np.inf
        arg_max = 0
        # use greedy search to find the index
        for i in delta_r.keys():
            target_value = delta_r[i] + args.lam_da*delta_g[i]
            if target_value > max_value:
                max_value = target_value
                arg_max = i
        # add new recommendation to set s
        s.append(arg_max)
        size_s = len(s)
        c = np.linalg.inv(
            np.matmul(np.transpose(features[:, s]), features[:, s])
            + sigma**2 * np.identity(size_s)
        )
    return s












