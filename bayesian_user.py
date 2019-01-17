import argparse
import os
import numpy as np
from utils import logging
from collections import defaultdict
import pdb
from ddp import fast_greedy_map, fast_window_map_dpp
import time


def parameter_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_log', default=False)
    parser.add_argument('--random_state', default=0)
    parser.add_argument('--verbose', default=2)
    parser.add_argument('--lam_da', default=10.0)
    parser.add_argument('--sigma', default=1.)
    parser.add_argument('--num_recommendation', default=5)
    parser.add_argument('--hidden_dim', default=10)
    parser.add_argument('--user_dim', default=4831)
    parser.add_argument('--movie_dim', default=3497)
    parser.add_argument('--num_bandit_iter', default=20)
    parser.add_argument('--dpp_theta', default=0.5)
    parser.add_argument('--dpp_w_size', default=5)
    parser.add_argument('--N', default=3497.)
    return parser.parse_args()


def f_user_ratings_new():
    file_name = "ml_1m_user/ml_1m_user_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def f_alpha(d, t, m, lam_da, sigma, s):
    # return np.sqrt(d*np.log((1+t*m/lam_da)/sigma)) + np.sqrt(lam_da)*s
    return 1.


def rho(x):
    return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))


def lam_da(x):
    return 1./(4*x)*np.tanh(x/2.)


def miu(c, x):
    part_a = ((1+c)-np.exp(-2*x))/((1+c)+2*np.exp(-x)+np.exp(-2*x))
    part_b = (-2*np.exp(-x)-2*np.exp(-2*x))/(1+2*np.exp(-x)+np.exp(-2*x))
    return 1./(2*x)*(part_a-part_b)


def bayesian_dpp(embeddings, new_index, test_items, args,
                 num=10, lamb_da=100):
    """
        user_emb: user embedding, shape (d, 1)
        movie_embs: movie embeddings, shape (d, m)
    """
    hidden_dim, num_movies = embeddings.shape
    matrix_s = lamb_da * np.identity(hidden_dim, dtype=np.float32)
    vector_m = np.zeros(shape=hidden_dim, dtype=np.float32)
    prec = []
    prec_all = []
    s_all = []
    
    # feature normalization
    nor_embs = embeddings.T.copy()
    for i in range(num_movies):
        nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
    
    for t in range(num):
        # theta_hat = np.random.multivariate_normal(vector_m, matrix_s)
        theta_hat = vector_m
        # assert theta_hat.shape == vector_m.shape
        p_hat = rho(np.dot(theta_hat, embeddings))
        
        # get recommendation set s
        s_inx = fast_greedy_map(p_hat, nor_embs, args.num_recommendation, args.dpp_theta)

        x = embeddings[:, np.array(s_inx)]
        # x = nor_embs[:, np.array(s_inx)]
        m = vector_m.reshape(hidden_dim, 1)
        xi_tmp = np.dot(x.T, matrix_s+np.dot(m, m.T))            # shape=(m_s, d)
        xi = np.sqrt(np.sum(xi_tmp.T*x, axis=0))                 # shape=(m_s)
        eta_tmp = np.dot(embeddings.T, matrix_s + np.dot(m, m.T))    # shape=(m, d)
        eta = np.sqrt(np.sum(eta_tmp.T * embeddings, axis=0))      # shape=(m)
        
        inv_matrix_s = np.linalg.inv(matrix_s)
        lam_xi = np.tile(lam_da(xi), (hidden_dim, 1))
        # assert lam_xi.shape == (hidden_dim, args.num_recommendation)
        s_i = np.sum((lam_xi*x).T[:, None, :]*x.T[:, :, None], axis=0)
        
        const = np.sum(embeddings * embeddings, axis=0)
        # assert const.shape == (num_movies, )
        miu_eta = np.tile(miu(const, eta), (hidden_dim, 1))
        # assert miu_eta.shape == (hidden_dim, num_movies)
        s_j = np.sum((miu_eta * embeddings).T[:, None, :] * embeddings.T[:, :, None], axis=0)
        
        inv_matrix_s_post = inv_matrix_s + 2*s_i  # + 2*s_j/args.N
        matrix_s = np.linalg.inv(inv_matrix_s_post)
        # assert matrix_s.shape == (hidden_dim, hidden_dim)
        reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
        
        m_i = np.sum(np.tile((reward+3./2), (hidden_dim, 1))*x, axis=1)
        # assert m_i.shape == (hidden_dim, )
        m_j = np.sum(embeddings, axis=1)
        # assert m_j.shape == (hidden_dim, )
        vector_m_post = np.dot(inv_matrix_s, vector_m)+m_i
        vector_m = np.dot(matrix_s, vector_m_post)
        # assert vector_m.shape == (hidden_dim, )

        # compute precision
        s_new = s_inx.copy()
        for i in range(args.dpp_w_size):
            s_new[i] = new_index[s_inx[i]]
            s_all.append(s_new[i])
        inter_set = set(s_new).intersection(set(list(test_items.keys())))
        prec_curr = float(len(inter_set)) / float(args.num_recommendation)
        prec.append(prec_curr)
        
        inter_set_all = set(s_all).intersection(set(list(test_items.keys())))
        prec_curr_all = float(len(inter_set_all)) / float(
            args.num_recommendation*args.num_bandit_iter)
        prec_all.append(prec_curr_all)
        
        embeddings = np.delete(embeddings, s_inx, 1)
        nor_embs = np.delete(nor_embs, s_inx, 0)
        new_index = np.delete(new_index, s_inx)
        
    return np.array(prec)


if __name__ == '__main__':
    args = parameter_setting()
    # logging
    if args.is_log:
        file_name = os.path.basename(__file__)
        output_path = logging(file_name, verbose=2)

    test_user_ratings = f_user_ratings_new()
    user_embs = np.load("ml_1m_user/ml_1m_user_emb_10.npy").T
    movie_embs = np.load("ml_1m_user/ml_1m_movie_emb_10.npy").T
    sim_mat = np.dot(movie_embs.T, movie_embs)
    
    for theta in [0.9]:
        test_precision = np.zeros(args.num_bandit_iter)
        args.dpp_theta = theta
        t1 = time.clock()
        length = 0.
        for user in test_user_ratings.keys():
            index = np.arange(0, args.movie_dim)
            if len(test_user_ratings[user]) > 150:
                mv_embs = movie_embs.copy()
                prec = bayesian_dpp(mv_embs, index,
                                    test_user_ratings[user], args, num=args.num_bandit_iter)
                print(user)
                print(prec)
                test_precision += prec
                length += 1
            
        print("lambda:{0}, test_precision:{1}".format(args.lam_da, test_precision / length))
        print("time used:%s" % (time.clock() - t1))
    

