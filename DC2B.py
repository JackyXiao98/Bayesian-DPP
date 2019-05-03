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
    parser.add_argument('--is_log', default=False)
    parser.add_argument('--random_state', default=0)
    parser.add_argument('--verbose', default=2)
    parser.add_argument('--lam_da', default=1)
    parser.add_argument('--sigma', default=1.)
    parser.add_argument('--num_recommendation', default=10)
    parser.add_argument('--hidden_dim', default=10)
    parser.add_argument('--user_dim', default=4831)
    parser.add_argument('--movie_dim', default=3496)
    parser.add_argument('--num_bandit_iter', default=10)
    parser.add_argument('--dpp_theta', default=0.5)
    parser.add_argument('--dpp_w_size', default=10)
    return parser.parse_args()


def f_train_ratings():
    file_name = "ml_1m_user_new/ml-1m_user_0.8_train.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def f_user_ratings():
    file_name = "ml_1m_user_new/ml-1m_user_0.8_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def rho(x):
    return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))


def lam_da(x):
    return 1./(4*x)*np.tanh(x/2.)


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


def bayes_greedy_map(scores, movie_embs, K, theta):
    """
        all_movie_embs: m * d
    """
    C = defaultdict(list)
    # alpha = theta / (1 - theta)
    alpha = theta
    # vec = np.sqrt(rho(scores))
    vec = np.exp(alpha * scores)

    D = vec * vec
    D = D * np.sum(movie_embs * movie_embs, axis=1)
    j = np.argmax(D * rho(scores))
    Y = [j]  # the selected set
    m = scores.shape[0]
    Z = set(range(m))  # the remained items
    for k in range(1, K):
        Z = Z - set(Y)
        for i in Z:
            # define similarity matrix
            Sji = (np.sum(movie_embs[j] * movie_embs[i]) + 1) / 2
            # define L kernel
            Lji = vec[j] * vec[i] * Sji
            # Lji = Sji
        
            if len(C[i]) == 0 or len(C[j]) == 0:
                ei = Lji / (D[j] ** 0.5)
            else:
                ei = (Lji - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j] ** 0.5)
            C[i].append(ei)
            D[i] = D[i] - ei ** 2
        ii = np.array(list(Z))
        # greedy search for the next item
        jj = np.argmax(D[ii] * rho(scores[ii]))
        j = ii[jj]
        Y.append(j)
    # pdb.set_trace()
    return Y


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


def bayesian_dpp(embeddings, test_items, args,
                 num=10, lamb_da=0.1,
                 train_items=None, user_emb=None, cate_sim=None):
    """
        user_emb: user embedding, shape (d, 1)
        movie_embs: movie embeddings, shape (d, m)
    """
    hidden_dim, num_movies = embeddings.shape
    matrix_s = lamb_da * np.identity(hidden_dim, dtype=np.float32)
    vector_m = np.ones(shape=hidden_dim, dtype=np.float32)
    
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
    nor_embs = embeddings.T.copy()
    for i in range(num_movies):
        nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
    
    for t in range(num):
        theta_hat = vector_m
        p_hat = np.dot(theta_hat, embeddings)

        can_items = can_items - set(rec_items)
        kk = np.array(list(can_items))

        if t == 0 and user_emb is not None:
            s_inx = mf_recommendation(user_emb, embeddings, can_items, size=5)
        else:
            # get recommendation set s then delete it from the candidate items
            s_tmp = bayes_greedy_map(p_hat[kk], nor_embs[kk, :], args.num_recommendation, args.dpp_theta)
            s_inx = [kk[i] for i in s_tmp]
        rec_items.extend(s_inx)

        x = embeddings[:, np.array(s_inx)]
        m = vector_m.reshape(hidden_dim, 1)
        xi_tmp = np.dot(x.T, matrix_s+np.dot(m, m.T))            # shape=(m_s, d)
        xi = np.sqrt(np.sum(xi_tmp.T*x, axis=0))                 # shape=(m_s)

        inv_matrix_s = np.linalg.inv(matrix_s)
        lam_xi = np.tile(lam_da(xi), (hidden_dim, 1))
        s_i = np.sum((lam_xi*x).T[:, None, :]*x.T[:, :, None], axis=0)
        
        # here we get updated covariance matrix S
        inv_matrix_s_post = inv_matrix_s + 2*s_i
        matrix_s = np.linalg.inv(inv_matrix_s_post)
        
        # here we get our reward Y
        reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
        
        m_i = np.sum(np.tile((reward+3./2), (hidden_dim, 1))*x, axis=1)
        
        # here we get updated mean vector m
        vector_m_post = np.dot(inv_matrix_s, vector_m)+m_i
        vector_m = np.dot(matrix_s, vector_m_post)
        
        # precision for each iteration
        inter_set = set(s_inx).intersection(set(list(test_items.keys())))
        prec_curr = float(len(inter_set)) / float(args.num_recommendation)
        prec.append(prec_curr)
        
        # compute recall
        s_test = set(list(test_items.keys()))
        recall_curr = float(len(inter_set)) / float(len(s_test))
        recall.append(recall_curr)
        
        # calculate the diversity
        div_curr = diversity(s_inx, embeddings)
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
        print(args)

    test_user_ratings = f_user_ratings()
    train_user_ratings = f_train_ratings()
    movie_cate_sim = np.load("ml_1m_0.8/ml-1m_tmp_0.8_10_item_sim.npy")
    user_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_user_embs.npy").T
    movie_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_item_embs.npy").T
    args.user_dim = user_embs.shape[1]
    args.movie_dim = movie_embs.shape[1]
    sim_mat = np.dot(movie_embs.T, movie_embs)

    for theta in [3]:
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
                prec, rec, div, cate_div = bayesian_dpp(mv_embs, test_user_ratings[user], args,
                                                        num=args.num_bandit_iter, lamb_da=l,
                                                        train_items=None,
                                                        user_emb=None,
                                                        cate_sim=movie_cate_sim)
                print(prec)
                test_precision += prec
                test_recall += rec
                test_diversity += div
                test_cate_diversity += cate_div
                length += 1
            else:
                mv_embs = movie_embs.copy()
                prec, rec, div, cate_div = bayesian_dpp(mv_embs, test_user_ratings[user], args,
                                                        num=args.num_bandit_iter, lamb_da=l,
                                                        train_items=None,
                                                        user_emb=None,
                                                        cate_sim=movie_cate_sim)
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
    

