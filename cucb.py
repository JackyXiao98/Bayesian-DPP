
import os
import pdb
import argparse
import time
import numpy as np
import xlwt

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def cosine_distances(matrix1, matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance


def load_history_dict(file_name):
    user_history = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'r') as inf:
        for line in inf:
            data = line.split("\t")
            u, i, r = data[0], data[1], data[2]
            user_history[int(u)][int(i)] = float(r)
    return user_history


def load_data(args):
    Tr = load_history_dict(args.train_file)
    Te = load_history_dict(args.test_file)
    user_embs = np.load(args.user_emb_file)
    movie_embs = np.load(args.movie_emb_file)
    return user_embs, movie_embs, Tr, Te 


def f_alpha(d, t, m, lam_da, sigma, s):
    # return np.sqrt(d*np.log((1+t*m/lam_da)/sigma)) + np.sqrt(lam_da)*s
    return 1.


def greedy_oracle(scores, can_items, sim, k, sigma, lam_da):
    """
        sim = X.T * X
    """
    kk = np.array(list(can_items))
    i = np.argmax(scores[kk])
    S = [kk[i]] # the first item
    C = 1.0 / (sim[i, i] + sigma**(2))

    for j in range(1, k):
        c_set = can_items - set(S)
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


def online_result(movie_embs, user_emb, id_action):
    theta = 0.8  # diversity rate TODO
    score_1 = None
    result = []

    for v in id_action:
        if np.random.rand() <= 0.5:
            score = np.random.rand()
        else:
            if score_1 is None:
                score = sigmoid(user_emb.dot(movie_embs[v]))
            else:
                relevance = sigmoid(user_emb.dot(movie_embs[v]))
                similarity = (cosine_distances(score_1, movie_embs[v][np.newaxis, :]) + 1) / 2
                similarity = np.mean(similarity)
                score = theta*relevance + (1-theta)*similarity

        if score >= 0.7:
            if score_1 is None:
                score_1 = movie_embs[v][np.newaxis, :]
            else:
                score_1 = np.append(score_1, values=movie_embs[v][np.newaxis, :], axis=0)
            result.append(1)
        else:
            result.append(0)

    return np.array(result)


def c2ucb(user_emb, movie_embs, train_items, test_items, args, num, sim=None, lamb_da=100):
    """
        movie_embs: movie embeddings, shape (d, m)
    """
    hidden_dim, num_movies = movie_embs.shape
    matrix_v = lamb_da*np.identity(hidden_dim, dtype=np.float32)
    vector_b = np.zeros(shape=hidden_dim, dtype=np.float32)
    prec, recall, div, total_reward_vec = [], [], [], []
    rec_items = []
    all_items = set(range(num_movies))

    can_items = all_items - set(train_items)
    cur_total_reward = 0

    for t in range(num):

        alpha_t = f_alpha(d=hidden_dim, t=t, m=num_movies, sigma=args.sigma,
                            lam_da=args.lam_da, s=1)
        inv_matrix_v = np.linalg.inv(matrix_v)
        theta_hat = np.dot(inv_matrix_v, vector_b)

        # compute the rating scores
        r_bar = np.dot(theta_hat, movie_embs)
        r_hat = np.dot(movie_embs.T, inv_matrix_v)
        r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar

        can_items = can_items - set(rec_items)

        # get recommendation set s
        s_inx = greedy_oracle(r_hat, can_items, sim, args.num_recommendation, args.sigma, args.lam_da)
        rec_items.extend(s_inx)

        x = movie_embs[:, np.array(s_inx)]
        matrix_v = matrix_v + np.dot(x, x.T)

        if args.off_line_eval:
            # for offline test
            reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
        else:
            # for online test
            reward = online_result(movie_embs.T, user_emb, s_inx)

        cur_total_reward += np.sum(reward)

        vector_b = vector_b + np.dot(x, reward)

        #compute precision
        s_test = set(list(test_items.keys()))
        inter_set = set(s_inx).intersection(s_test)
        prec_curr = float(len(inter_set)) / float(args.num_recommendation)
        prec.append(prec_curr)

        # compute recall
        recall_curr = float(len(inter_set)) / float(len(s_test))        
        recall.append(recall_curr)

        # compute diversity
        div_curr = diversity(s_inx, movie_embs)
        div.append(div_curr)

        total_reward_vec.append(cur_total_reward / float(len(s_test)))

    return np.array(prec), np.array(recall), np.array(div), np.array(total_reward_vec)


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


def eval(args):
    user_embs, movie_embs, Tr, Te = load_data(args)
    sim_mat = np.dot(movie_embs, movie_embs.T)
    lamda_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # write date to excel
    file = xlwt.Workbook(encoding='ascii')
    table = file.add_sheet('cucb')
    row0 = list(range(0, args.num_bandit_iter, 1))

    for i in range(len(lamda_list)):
        test_precision = np.zeros(args.num_bandit_iter)
        test_recall = np.zeros(args.num_bandit_iter)
        test_div = np.zeros(args.num_bandit_iter)
        test_reward = np.zeros(args.num_bandit_iter)
        args.lam_da = lamda_list[i]
        t1 = time.clock()

        for user in Te.keys():
            prec, recall, div, reward = c2ucb(user_embs[user], movie_embs.T, Tr[user], Te[user], args, num=args.num_bandit_iter, sim=sim_mat)
            print(prec, recall, div, reward)
            test_precision += prec
            test_recall += recall
            test_div += div
            test_reward += reward

        test_precision = test_precision / len(Te.keys())
        test_recall = test_recall / len(Te.keys())
        test_div = test_div / len(Te.keys())
        test_reward = test_reward / len(Te.keys())

        table.write(0 + 4*i, 0, 'process for lamda: '+str(lamda_list[i]))
        table.write(1 + 4*i, 0, 'precision')
        table.write(2 + 4*i, 0, 'diversity')
        table.write(3 + 4 * i, 0, 'recall')
        for j in range(len(row0)):
            table.write(0 + 4 * i, j + 1, row0[j] + 1)
            table.write(1 + 4 * i, j + 1, test_precision[j])
            table.write(2 + 4 * i, j + 1, test_div[j])
            table.write(3 + 4 * i, j + 1, test_recall[j])

        print("lambda:{0}\ntest_precision:{1}\ntest_recall:{2}\ntest_div:{3}\ntest_reward:{4}".format(args.lam_da, test_precision, test_recall, test_div, test_reward))
        print("time used:%s\n" % (time.clock() - t1))

    file.save('cucb result.xlsx')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="c2ucb for recommendation")

    parser.add_argument('--off_line_eval', default=1)
    parser.add_argument('--lam_da', default=10.0)
    parser.add_argument('--sigma', default=1.)
    parser.add_argument('--num_recommendation', default=5)
    parser.add_argument('--num_bandit_iter', default=20)

    # ml-1m data
    parser.add_argument('--train_file', default='./final-ml-1m/ml-1m_tmp_0.8_10_train.txt',
                        help='the training file')
    parser.add_argument('--test_file', default='./final-ml-1m/ml-1m_tmp_0.8_10_test.txt',
                        help='the testing file')
    parser.add_argument('--user_emb_file', default='./final-ml-1m/ml-1m_tmp_0.8_10_user_dim30_embs.npy',
                        help='the user embedding file')
    parser.add_argument('--movie_emb_file', default='./final-ml-1m/ml-1m_tmp_0.8_10_item_dim30_embs.npy',
                        help='the movie embedding file')

    # # ml-100k data
    # parser.add_argument('--train_file', default='../dataset/ml-100k1/ml-100k_train.txt', help='the training file')
    # parser.add_argument('--test_file', default='../dataset/ml-100k1/ml-100k_test.txt', help='the testing file')
    # parser.add_argument('--user_emb_file', default='../dataset/ml-100k1/ml100k_100_user_embs.npy',
    #                     help='the user embedding file')
    # parser.add_argument('--movie_emb_file', default='../dataset/ml-100k1/ml100k_100_item_embs.npy',
    #                     help='the movie embedding file')

    args = parser.parse_args()

    eval(args)
