import argparse
import os
import numpy as np
from utils import logging
from collections import defaultdict
import time
from sklearn.metrics.pairwise import cosine_similarity
import pdb


def parameter_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_log', default=True)
    parser.add_argument('--random_state', default=0)
    parser.add_argument('--verbose', default=2)
    parser.add_argument('--lam_da', default=0.1)
    parser.add_argument('--sigma', default=1.)
    parser.add_argument('--num_recommendation', default=5)
    parser.add_argument('--hidden_dim', default=10)
    parser.add_argument('--user_dim', default=4831)
    parser.add_argument('--movie_dim', default=3496)
    parser.add_argument('--num_bandit_iter', default=10)
    parser.add_argument('--dpp_theta', default=0.5)
    parser.add_argument('--dpp_w_size', default=5)
    return parser.parse_args()


def f_train_ratings():
    file_name = "ml_1m_0.8/ml-1m_tmp_0.8_10_train.txt"
    # file_name = "ml_1m/ml-1m_tmp_0.7_10_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def f_user_ratings():
    file_name = "ml_1m_user_new/ml-1m_user_0.8_test.txt"
    # file_name = "ml_1m_0.8/ml-1m_tmp_0.8_10_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def rho(x):
    return 1. / (1. + np.exp(-x))


def lam_da(x):
    return 1./(4*x)*np.tanh(x/2.)


def g_x(x, xi):
    return x/2. - xi/2. + np.log(rho(xi)) - lam_da(xi)*(x**2 - xi**2)


def q_x(x_c, x_e, xi_c, xi_e):
    tmp = np.exp(g_x(x_c, xi_c) + g_x(x_e, xi_e) - g_x(-x_e, xi_e))
    return tmp/(1.+tmp)


def diversity(s_inx, X):
    """
        s_inx: the selected item set, a numpy vector
        X: the item feature matrix, shape (d, m)
        The similarities between items are measured using cosine similarity.
    """
    S = cosine_similarity(X[:, s_inx].T)
    ii, jj = np.triu_indices(len(s_inx), k=1)
    vec = S[ii, jj]
    div = 1 - np.mean(vec)
    return div


def greedy_search(r_c, r_e):

    D = rho(r_c) * rho(r_e)
    j = np.argmax(D)
    return j


def mf_recommendation(user_emb, movie_embs, can_items, size=5):
    inx = np.array(list(can_items))
    scores = np.dot(user_emb, movie_embs[:, inx])
    ii = np.argsort(scores)[::-1][:size]
    return list(inx[ii])


def pre_diversity(s_inx, item_sim):
    num = len(s_inx)
    s = []
    for i in range(num):
        for j in range(i+1, num):
            s.append(1 - item_sim[s_inx[i], s_inx[j]])
    return sum(s) / len(s)


def nips(embs_c, embs_e, test_items, args,
         num=10, lamb_da=0.1,
         train_items=None, user_emb=None, cate_sim=None):
    """
        user_emb: user embedding, shape (d, 1)
        movie_embs: movie embeddings, shape (d, m)
    """
    hidden_dim, num_movies = embs_c.shape
    matrix_s_c = lamb_da * np.identity(hidden_dim, dtype=np.float32)
    vector_m_c = np.ones(shape=hidden_dim, dtype=np.float32)
    matrix_s_e = lamb_da * np.identity(hidden_dim, dtype=np.float32)
    vector_m_e = np.ones(shape=hidden_dim, dtype=np.float32)
    
    prec = []
    recall = []
    div = []
    cate_div = []
    rec_items = []
    all_items = set(range(num_movies))

    if train_items is None:
        can_items = all_items
    else:
        can_items = all_items - set(train_items)
    
    # feature normalization
    nor_embs_c = embs_c.T.copy()
    nor_embs_e = embs_e.T.copy()
    for i in range(num_movies):
        nor_embs_c[i, :] = nor_embs_c[i, :] / np.linalg.norm(nor_embs_c[i, :])
        nor_embs_e[i, :] = nor_embs_e[i, :] / np.linalg.norm(nor_embs_e[i, :])
    
    for t in range(num):
        s_inx = []
        for number in range(5):
            theta_hat_c = vector_m_c
            p_hat_c = np.dot(theta_hat_c, nor_embs_c.T)   # d * (d, m) = (m)
            theta_hat_e = vector_m_e
            p_hat_e = np.dot(theta_hat_e, nor_embs_e.T)   # d * (d, m) = (m)
    
            can_items = can_items - set(rec_items)
            kk = np.array(list(can_items))
    
            # if t == 0 and user_emb is not None:
            #     s_inx = mf_recommendation(user_emb, embs_c, can_items, size=5)
            # else:
            #     # get recommendation set s then delete it from the candidate items
            #     s_tmp = greedy_search(p_hat_c[kk], p_hat_e[kk], args.num_recommendation)
            #     s_inx = [kk[i] for i in s_tmp]
            # # print(s_inx)
            
            item_tmp = greedy_search(p_hat_c[kk], p_hat_e[kk])
            item = kk[item_tmp]
            rec_items.extend([item])
            s_inx.append(item)
    
            x_c = nor_embs_c.T[:, item]
            m_c = vector_m_c.reshape(hidden_dim, 1)
            xi_tmp_c = np.dot(x_c.T, matrix_s_c+np.dot(m_c, m_c.T))            # shape=(m_s, d)
            xi_c = np.sqrt(np.dot(xi_tmp_c, x_c))                     # shape=(m_s)
            x_e = nor_embs_e.T[:, item]
            m_e = vector_m_e.reshape(hidden_dim, 1)
            xi_tmp_e = np.dot(x_e.T, matrix_s_e+np.dot(m_e, m_e.T))            # shape=(m_s, d)
            xi_e = np.sqrt(np.dot(xi_tmp_e, x_e))                     # shape=(m_s)
    
            q = q_x(np.dot(vector_m_c, x_c), np.dot(vector_m_e, x_e), xi_c, xi_e)
            # here we get our reward Y
            reward = np.array([1.0 if item in test_items else 0.0])
            
            inv_matrix_s_c = np.linalg.inv(matrix_s_c)
            x_c_tmp = x_c.reshape(hidden_dim, 1)
            s_i_c = q**(1-reward) * lam_da(xi_c) * np.dot(x_c_tmp, x_c_tmp.T)
            inv_matrix_s_e = np.linalg.inv(matrix_s_e)
            x_e_tmp = x_e.reshape(hidden_dim, 1)
            s_i_e = lam_da(xi_e) * np.dot(x_e_tmp, x_e_tmp.T)
            
            # here we get updated covariance matrix S
            inv_matrix_s_post_c = inv_matrix_s_c + 2*s_i_c
            matrix_s_c = np.linalg.inv(inv_matrix_s_post_c)
            inv_matrix_s_post_e = inv_matrix_s_e + 2*s_i_e
            matrix_s_e = np.linalg.inv(inv_matrix_s_post_e)
            
            m_i_c = (-q)**(1-reward)*x_c
            m_i_e = (2*q-1) ** (1 - reward) * x_e

            # here we get updated mean vector m
            vector_m_post_c = np.dot(inv_matrix_s_c, vector_m_c)+1./2*m_i_c
            vector_m_c = np.dot(matrix_s_c, vector_m_post_c)
            vector_m_post_e = np.dot(inv_matrix_s_e, vector_m_e)+1./2*m_i_e
            vector_m_e = np.dot(matrix_s_e, vector_m_post_e)
        
        # precision for each iteration
        inter_set = set(s_inx).intersection(set(list(test_items.keys())))
        prec_curr = float(len(inter_set)) / float(args.num_recommendation)
        prec.append(prec_curr)
        
        # compute recall
        s_test = set(list(test_items.keys()))
        recall_curr = float(len(inter_set)) / float(len(s_test))
        recall.append(recall_curr)
        
        # calculate the diversity
        div_curr = diversity(s_inx, embs_c)
        div.append(div_curr)
        
        if cate_sim is None:
            cate_div_cur = 0.0
        else:
            cate_div_cur = pre_diversity(s_inx, cate_sim)
        cate_div.append(cate_div_cur)
        
    return np.array(prec), np.array(recall), np.array(div), np.array(cate_div)


if __name__ == '__main__':
    args = parameter_setting()
    # logging
    if args.is_log:
        file_name = os.path.basename(__file__)
        output_path = logging(file_name, verbose=2)

    test_user_ratings = f_user_ratings()
    # train_user_ratings = f_train_ratings()
    # user_embs = np.load("ml_1m/lmf_ml-1m_tmp_0.7_10_dim10_item_embs.npy").T
    # movie_embs = np.load("ml_1m/lmf_ml-1m_tmp_0.7_10_dim10_user_embs.npy").T
    # user_embs = np.load("ml_1m_0.8/lmf_ml-1m_tmp_0.8_10_dim10_user_embs.npy").T
    # movie_embs = np.load("ml_1m_0.8/lmf_ml-1m_tmp_0.8_10_dim10_item_embs.npy").T
    # movie_cate_sim = np.load("ml_1m_0.8/ml-1m_tmp_0.8_10_item_sim.npy")
    user_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_user_embs.npy").T
    movie_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_item_embs.npy").T
    args.user_dim = user_embs.shape[1]
    args.movie_dim = movie_embs.shape[1]
    sim_mat = np.dot(movie_embs.T, movie_embs)

    for theta in [4]:
        print(theta)
        l = 1.
        test_precision = np.zeros(args.num_bandit_iter)
        test_recall = np.zeros(args.num_bandit_iter)
        test_diversity = np.zeros(args.num_bandit_iter)
        test_cate_diversity = np.zeros(args.num_bandit_iter)
        
        test_precision_low = np.zeros(args.num_bandit_iter)
        test_recall_low = np.zeros(args.num_bandit_iter)
        test_diversity_low = np.zeros(args.num_bandit_iter)
        test_cate_diversity_low = np.zeros(args.num_bandit_iter)
        args.dpp_theta = theta
        t1 = time.clock()
        length = 0.
        length_low = 0.
        for user in test_user_ratings.keys():
            if len(test_user_ratings[user]) >= 20:
                print(user)
                mv_embs = movie_embs.copy()
                prec, rec, div, cate_div = nips(mv_embs, mv_embs, test_user_ratings[user], args,
                                                num=args.num_bandit_iter, lamb_da=l,
                                                train_items=None,
                                                user_emb=None,
                                                cate_sim=None)
                print(prec)
                # print(rec)
                # print(div)
                test_precision += prec
                test_recall += rec
                test_diversity += div
                test_cate_diversity += cate_div
                length += 1
            else:
                mv_embs = movie_embs.copy()
                prec, rec, div, cate_div = nips(mv_embs, mv_embs, test_user_ratings[user], args,
                                                num=args.num_bandit_iter, lamb_da=l,
                                                train_items=None,
                                                user_emb=None,
                                                cate_sim=None)
                print(user)
                print(prec)
                # print(rec)
                # print(div)
                test_precision_low += prec
                test_recall_low += rec
                test_diversity_low += div
                test_cate_diversity_low += cate_div
                length_low += 1
            
        print("theta:{0}, ".format(theta))
        print("test_precision:{0}".format(test_precision / length))
        print("test_recall:{0}".format(test_recall / length))
        print("test_diversity:{0}".format(test_diversity / length))
        print("test_cate_diversity:{0}".format(test_cate_diversity / length))
        print("test_precision_low:{0}".format(test_precision_low / length_low))
        print("test_precision_all:{0}".format((test_precision+test_precision_low)/(length+length_low)))
        print("time used:%s" % (time.clock() - t1))
    

