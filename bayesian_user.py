import argparse
import os
import numpy as np
from utils import logging
from collections import defaultdict
import time
from sklearn.metrics.pairwise import cosine_similarity


def parameter_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_log', default=False)
    parser.add_argument('--random_state', default=0)
    parser.add_argument('--verbose', default=2)
    parser.add_argument('--lam_da', default=0.1)
    parser.add_argument('--sigma', default=1.)
    parser.add_argument('--num_recommendation', default=5)
    parser.add_argument('--hidden_dim', default=10)
    parser.add_argument('--user_dim', default=5328)
    parser.add_argument('--movie_dim', default=3569)
    parser.add_argument('--num_bandit_iter', default=10)
    parser.add_argument('--dpp_theta', default=0.9)
    parser.add_argument('--dpp_w_size', default=5)
    parser.add_argument('--N', default=3543.)
    return parser.parse_args()


def f_user_ratings():
    file_name = "ml_1m_user/ml_1m_user_test.txt"
    # file_name = "ml_1m/ml-1m_tmp_0.7_10_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def f_train_ratings():
    file_name = "ml_1m_user/ml_1m_user_train.txt"
    # file_name = "ml_1m/ml-1m_tmp_0.7_10_test.txt"
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def f_user_ratings_new():
    # file_name = "ml_1m_user/ml_1m_user_test.txt"
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


def miu(c, x):
    part_a = ((1+c)-np.exp(-2*x))/((1+c)+2*np.exp(-x)+np.exp(-2*x))
    part_b = (-2*np.exp(-x)-2*np.exp(-2*x))/(1+2*np.exp(-x)+np.exp(-2*x))
    return 1./(2*x)*(part_a-part_b)


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
        movie_embs: m * d
    """
    C = defaultdict(list)
    alpha = theta / (1 - theta)
    vec = np.sqrt(rho(scores))
    vec = np.exp(alpha*scores)

    D = vec * vec
    D = D * np.sum(movie_embs * movie_embs, axis=1)
    j = np.argmax(D)
    Y = [j]               # the selected set
    m = scores.shape[0]
    Z = set(range(m))     # the remained items
    for k in range(1, K):
        Z = Z - set(Y)
        for i in Z:
            # define similarity matrix
            Sji = (np.sum(movie_embs[j] * movie_embs[i]) + 1) / 2
            # define L kernel
            Lji = vec[j] * vec[i] * Sji
            
            if len(C[i]) == 0 or len(C[j]) == 0:
                ei = Lji / (D[j] ** 0.5)
            else:
                ei = (Lji - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j] ** 0.5)
            C[i].append(ei)
            D[i] = D[i] - ei ** 2
        ii = np.array(list(Z))
        # greedy search for the next item
        jj = np.argmax(2*np.log(D)[ii]+np.log(rho(scores))[ii])
        j = ii[jj]
        Y.append(j)

    return Y


def bayesian_dpp(embeddings, new_index, test_items, args,
                 num=10, lamb_da=0.1):
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
    recall = []
    div = []
    
    # feature normalization
    nor_embs = embeddings.T.copy()
    for i in range(num_movies):
        nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
    
    for t in range(num):
        # theta_hat = np.random.multivariate_normal(vector_m, matrix_s)
        theta_hat = vector_m
        p_hat = np.dot(theta_hat, embeddings)
        
        # get recommendation set s
        s_inx = bayes_greedy_map(p_hat, nor_embs, args.num_recommendation, args.dpp_theta)

        x = embeddings[:, np.array(s_inx)]
        # x = nor_embs[:, np.array(s_inx)]
        m = vector_m.reshape(hidden_dim, 1)
        xi_tmp = np.dot(x.T, matrix_s+np.dot(m, m.T))            # shape=(m_s, d)
        xi = np.sqrt(np.sum(xi_tmp.T*x, axis=0))                 # shape=(m_s)
        # eta_tmp = np.dot(embeddings.T, matrix_s + np.dot(m, m.T))    # shape=(m, d)
        # eta = np.sqrt(np.sum(eta_tmp.T * embeddings, axis=0))      # shape=(m)
        
        inv_matrix_s = np.linalg.inv(matrix_s)
        lam_xi = np.tile(lam_da(xi), (hidden_dim, 1))
        s_i = np.sum((lam_xi*x).T[:, None, :]*x.T[:, :, None], axis=0)
        
        # const = np.sum(embeddings * embeddings, axis=0)
        # miu_eta = np.tile(miu(const, eta), (hidden_dim, 1))
        # s_j = np.sum((miu_eta * embeddings).T[:, None, :] * embeddings.T[:, :, None], axis=0)
        
        # here we get updated covariance matrix S
        inv_matrix_s_post = inv_matrix_s + 2*s_i  # + 2*s_j/args.N
        matrix_s = np.linalg.inv(inv_matrix_s_post)
        
        # here we get our reward Y
        reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
        
        m_i = np.sum(np.tile((reward+3./2), (hidden_dim, 1))*x, axis=1)
        # m_j = np.sum(embeddings, axix`s=1)
        
        # here we get updated mean vector m
        vector_m_post = np.dot(inv_matrix_s, vector_m)+m_i
        vector_m = np.dot(matrix_s, vector_m_post)

        # compute precision
        s_new = s_inx.copy()
        # reindex our set because we delete some items
        for i in range(args.dpp_w_size):
            s_new[i] = new_index[s_inx[i]]
            s_all.append(s_new[i])
        
        # precision for each iteration
        inter_set = set(s_new).intersection(set(list(test_items.keys())))
        prec_curr = float(len(inter_set)) / float(args.num_recommendation)
        prec.append(prec_curr)
        
        # mean precision for all iteration
        inter_set_all = set(s_all).intersection(set(list(test_items.keys())))
        prec_curr_all = float(len(inter_set_all)) / float(
            args.num_recommendation*args.num_bandit_iter)
        prec_all.append(prec_curr_all)
        
        # compute recall
        s_test = set(list(test_items.keys()))
        recall_curr = float(len(inter_set)) / float(len(s_test))
        recall.append(recall_curr)
        
        # calculate the diversity
        div_curr = diversity(s_inx, embeddings)
        div.append(div_curr)
        
        # delete the recommended itmes
        embeddings = np.delete(embeddings, s_inx, 1)
        nor_embs = np.delete(nor_embs, s_inx, 0)
        new_index = np.delete(new_index, s_inx)
        
    return np.array(prec), np.array(recall), np.array(div)


def base_greedy_map(scores, movie_embs, K, theta):
    """
        movie_embs: m * d
    """
    C = defaultdict(list)
    alpha = theta / (1 - theta)
    vec = rho(scores)
    
    D = vec * vec
    D = D * np.sum(movie_embs * movie_embs, axis=1)
    j = np.argmax(D)
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
            
            if len(C[i]) == 0 or len(C[j]) == 0:
                ei = Lji / (D[j] ** 0.5)
            else:
                ei = (Lji - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j] ** 0.5)
            C[i].append(ei)
            D[i] = D[i] - ei ** 2
        ii = np.array(list(Z))
        # greedy search for the next item
        jj = np.argmax(D[ii])
        j = ii[jj]
        Y.append(j)
    
    return Y


def c2ucb(movie_embs, test_items, args, num=10, lamb_da=100):
    hidden_dim, num_movies = movie_embs.shape
    matrix_v = lamb_da * np.identity(hidden_dim, dtype=np.float32)
    vector_b = np.zeros(shape=hidden_dim, dtype=np.float32)
    new_index = np.arange(0, args.movie_dim)
    prec = []
    prec_all = []
    s_all = []
    
    # feature normalization
    nor_embs = movie_embs.T.copy()
    for i in range(num_movies):
        nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
    
    for t in range(num):
        alpha_t = 1.
        
        inv_matrix_v = np.linalg.inv(matrix_v)
        
        theta_hat = np.dot(inv_matrix_v, vector_b)
        
        # compute the rating scores
        r_bar = np.dot(theta_hat, movie_embs)
        r_hat = np.dot(movie_embs.T, inv_matrix_v)
        r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar
        
        # get recommendation set s
        sim = np.dot(movie_embs.T, movie_embs)
        s_inx = bayes_greedy_map(r_hat, nor_embs, args.num_recommendation, args.dpp_theta)
        
        x = movie_embs[:, np.array(s_inx)]
        
        matrix_v = matrix_v + np.dot(x, x.T)
        
        reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
        
        vector_b = vector_b + np.dot(x, reward)
        
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
            args.num_recommendation * args.num_bandit_iter)
        prec_all.append(prec_curr_all)
        
        movie_embs = np.delete(movie_embs, s_inx, 1)
        nor_embs = np.delete(nor_embs, s_inx, 0)
        new_index = np.delete(new_index, s_inx)
    
    return np.array(prec)


def max_dic_size(d, threshold):
    count = 0
    for key in d.keys():
        length = len(d[key])
        if length > threshold:
            count += 1
            
    return count


if __name__ == '__main__':
    args = parameter_setting()
    # logging
    if args.is_log:
        file_name = os.path.basename(__file__)
        output_path = logging(file_name, verbose=2)

    test_user_ratings = f_user_ratings_new()
    # train_user_ratings = f_train_ratings()
    # user_embs = np.load("ml_1m_user/ml_1m_user_emb_10.npy").T
    # movie_embs = np.load("ml_1m_user/ml_1m_movie_emb_10.npy").T
    user_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_user_embs.npy").T
    movie_embs = np.load("ml_1m_user_new/bpr_ml-1m_user_0.8_dim10_item_embs.npy").T
    sim_mat = np.dot(movie_embs.T, movie_embs)
    
    for theta in [0.9, 0.95, 0.95]:
        print(theta)
        l = 1.
        test_precision = np.zeros(args.num_bandit_iter)
        test_recall = np.zeros(args.num_bandit_iter)
        test_diversity = np.zeros(args.num_bandit_iter)
        
        test_precision_low = np.zeros(args.num_bandit_iter)
        test_recall_low = np.zeros(args.num_bandit_iter)
        test_diversity_low = np.zeros(args.num_bandit_iter)
        args.dpp_theta = theta
        t1 = time.clock()
        length = 0.
        length_low = 0.
        for user in test_user_ratings.keys():
            index = np.arange(0, args.movie_dim)
            if len(test_user_ratings[user]) >= 20:
                mv_embs = movie_embs.copy()
                # prec = c2ucb(movie_embs, test_user_ratings[user], args,
                #              num=args.num_bandit_iter)
                # rec, div = 0., 0.
                prec, rec, div = bayesian_dpp(mv_embs, index,
                                    test_user_ratings[user], args,
                                    num=args.num_bandit_iter, lamb_da=l)
                print(user)
                print(prec)
                # print(rec)
                # print(div)
                test_precision += prec
                test_recall += rec
                test_diversity += div
                length += 1
            # else:
            #     mv_embs = movie_embs.copy()
            #     prec, rec, div = bayesian_dpp(mv_embs, index,
            #                                   test_user_ratings[user], args,
            #                                   num=args.num_bandit_iter, lamb_da=l)
            #     print(user)
            #     print(prec)
            #     # print(rec)
            #     # print(div)
            #     test_precision_low += prec
            #     test_recall_low += rec
            #     test_diversity_low += div
            #     length_low += 1
            
        print("theta:{0}, ".format(theta))
        print("test_precision:{0}".format(test_precision / length))
        # print("test_recall:{0}".format(test_recall / length))
        # print("test_diversity:{0}".format(test_diversity / length))
        print("test_precision_low:{0}".format(test_precision_low / length_low))
        print("test_precision_all:{0}".format((test_precision+test_precision_low)/(length+length_low)))
        print("time used:%s" % (time.clock() - t1))
    

