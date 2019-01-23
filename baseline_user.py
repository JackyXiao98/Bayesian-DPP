import argparse
import os
import numpy as np
from greedy_search import oracle, greedy_oracle
from utils import logging
from collections import defaultdict
from ddp import fast_greedy_map, fast_window_map_dpp
import time


def parameter_setting():
	parser = argparse.ArgumentParser()
	parser.add_argument('--is_log', default=False)
	parser.add_argument('--random_state', default=0)
	parser.add_argument('--verbose', default=2)
	parser.add_argument('--lam_da', default=1.0)
	parser.add_argument('--sigma', default=1.)
	parser.add_argument('--num_recommendation', default=5)
	parser.add_argument('--hidden_dim', default=10)
	parser.add_argument('--user_dim', default=4831)
	parser.add_argument('--movie_dim', default=3497)
	parser.add_argument('--num_bandit_iter', default=10)
	parser.add_argument('--dpp_theta', default=0.5)
	parser.add_argument('--dpp_w_size', default=5)
	parser.add_argument('--N', default=3497.)
	return parser.parse_args()


def load_test_data():
	"""
	load movie lens 100k ratings from original rating file.
	need to download and put rating data in /data folder first.
	Source: http://www.grouplens.org/
	"""
	file_path = 'test.dat'
	user_history = defaultdict(lambda: defaultdict(int))
	for line in open(file_path, 'r'):  # 打开指定文件
		(userid, movieid, rating, ts) = line.split('::')  # 数据集中每行有4项
		uid = int(userid)
		mid = int(movieid)
		rat = float(rating)
		user_history[int(uid)][int(mid)] = float(rat)
	
	return user_history


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


def fd_sketching(St, sketch_m, num_rec, lamb_da):
	u, s, vh = np.linalg.svd(St, full_matrices=False)
	m = sketch_m + num_rec - 1
	rho1 = np.zeros(m)
	rho2 = np.zeros(m)
	for i in range(m):
		if i < sketch_m - 1:
			rho1[i] = np.sqrt(s[i] ** 2 - s[m - 1] ** 2)
			rho2[i] = 1.0 / (s[i] ** 2 - s[m - 1] ** 2 + lamb_da)
		else:
			rho2[i] = 1.0 / lamb_da
	# pdb.set_trace()
	return rho1.reshape(m, 1) * vh, rho2.reshape(m, 1)


def rho(x):
	return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))


def c2ucb(movie_embs, test_items, args, num=10, lamb_da=100, sketch_m=5):
	"""
		user_emb: user embedding, shape (d, 1)
		movie_embs: movie embeddings, shape (d, m)
	"""
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
		alpha_t = f_alpha(d=hidden_dim, t=t, m=num_movies, sigma=args.sigma,
					lam_da=args.lam_da, s=1)

		inv_matrix_v = np.linalg.inv(matrix_v)
		
		theta_hat = np.dot(inv_matrix_v, vector_b)
		
		# compute the rating scores
		r_bar = np.dot(theta_hat, movie_embs)
		r_hat = np.dot(movie_embs.T, inv_matrix_v)
		r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar
		
		# get recommendation set s
		sim = np.dot(movie_embs.T, movie_embs)
		s_inx = greedy_oracle(r_hat, sim, args.num_recommendation, args.sigma, args.lam_da)
		
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


def diversity():
	pass


if __name__ == '__main__':
	args = parameter_setting()
	# logging
	if args.is_log:
		file_name = os.path.basename(__file__)
		output_path = logging(file_name, verbose=2)
	
	test_user_ratings = f_user_ratings_new()
	# user_embs, movie_embs = f_features_new()
	
	user_embs = np.load("ml_1m_user/ml_1m_user_emb_10.npy").T
	movie_embs = np.load("ml_1m_user/ml_1m_movie_emb_10.npy").T
	# user_embs = np.load("ml-100k/ml100k_10_user_embs.npy").T
	# movie_embs = np.load("ml-100k/ml100k_10_item_embs.npy").T
	
	
	# pdb.set_trace()
	
	sim_mat = np.dot(movie_embs.T, movie_embs)
	# pdb.set_trace()
	for l in [0.1, 0.5, 1, 10]:
		test_precision = np.zeros(args.num_bandit_iter)
		args.lam_da = l
		t1 = time.clock()
		length = 0.
		for user in test_user_ratings.keys():
			# prec = c2ucb_dpp_sketched(user_embs[:, user], movie_embs, test_user_ratings[user], args, num=args.num_bandit_iter)
			if len(test_user_ratings[user]) > 20:
				prec = c2ucb(movie_embs, test_user_ratings[user], args,
							num=args.num_bandit_iter)
				print(user)
				print(prec)
				test_precision += prec
				length += 1
		
		print("lambda:{0}, ".format(l))
		print("test_precision:{0}".format(test_precision / length))
		print("time used:%s" % (time.clock() - t1))


