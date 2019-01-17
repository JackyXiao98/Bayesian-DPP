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
	parser.add_argument('--lam_da', default=10.0)
	parser.add_argument('--sigma', default=1.)
	parser.add_argument('--num_recommendation', default=5)
	parser.add_argument('--hidden_dim', default=10)
	parser.add_argument('--user_dim', default=5331)
	parser.add_argument('--movie_dim', default=3468)
	parser.add_argument('--num_bandit_iter', default=20)
	parser.add_argument('--dpp_theta', default=0.5)
	parser.add_argument('--dpp_w_size', default=5)
	parser.add_argument('--N', default=3468.)
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


def c2ucb(user_emb, movie_embs, test_items, args, num=10, sim=None, lamb_da=100):
	"""
		user_emb: user embedding, shape (d, 1)
		movie_embs: movie embeddings, shape (d, m)
	"""
	hidden_dim, num_movies = movie_embs.shape
	matrix_v = lamb_da * np.identity(hidden_dim, dtype=np.float32)
	vector_b = np.zeros(shape=hidden_dim, dtype=np.float32)
	prec = []
	for t in range(num):
		alpha_t = f_alpha(d=hidden_dim, t=t, m=num_movies, sigma=args.sigma,
		                  lam_da=args.lam_da, s=1)
		inv_matrix_v = np.linalg.inv(matrix_v)
		theta_hat = np.dot(inv_matrix_v, vector_b)
		
		# if t > 0:
		#     theta_hat = np.dot(inv_matrix_v, vector_b)
		# else:
		#     theta_hat = user_emb
		
		# compute the rating scores
		r_bar = np.dot(theta_hat, movie_embs)
		r_hat = np.dot(movie_embs.T, inv_matrix_v)
		r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar
		
		# get recommendation set s
		# s_inx = oracle(args, r_hat, movie_embs)
		s_inx = greedy_oracle(r_hat, sim, args.num_recommendation, args.sigma, args.lam_da)
		
		x = movie_embs[:, np.array(s_inx)]
		matrix_v = matrix_v + np.dot(x, x.T)
		# print(s_inx)
		
		# r = [test_items[i] if i in test_items else 0 for i in s_inx]
		reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
		# pdb.set_trace()
		
		vector_b = vector_b + np.dot(x, reward)
		# pdb.set_trace()
		# compute precision
		inter_set = set(s_inx).intersection(set(list(test_items.keys())))
		prec_curr = float(len(inter_set)) / float(args.num_recommendation)
		# print(t, prec_curr)
		prec.append(prec_curr)
	
	return np.array(prec)


def c2ucb_dpp(user_emb, movie_embs, test_items, args, num=10, lamb_da=100, sketch_m=5):
	"""
		user_emb: user embedding, shape (d, 1)
		movie_embs: movie embeddings, shape (m, d)
	"""
	hidden_dim, num_movies = movie_embs.shape
	matrix_v = lamb_da * np.identity(hidden_dim, dtype=np.float32)
	vector_b = np.zeros(shape=hidden_dim, dtype=np.float32)
	prec = []
	
	# feature normalization
	nor_embs = movie_embs.T.copy()
	for i in range(num_movies):
		nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
	
	for t in range(num):
		alpha_t = f_alpha(d=hidden_dim, t=t, m=num_movies, sigma=args.sigma,
		                  lam_da=args.lam_da, s=1)
		
		# t1 = time.clock()
		inv_matrix_v = np.linalg.inv(matrix_v)
		# print("1 time used:%s" % (time.clock() - t1))
		
		theta_hat = np.dot(inv_matrix_v, vector_b)
		
		# compute the rating scores
		r_bar = np.dot(theta_hat, movie_embs)
		r_hat = np.dot(movie_embs.T, inv_matrix_v)
		r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar
		
		# get recommendation set s
		# t1 = time.clock()
		s_inx = fast_greedy_map(r_hat, nor_embs, args.num_recommendation, args.dpp_theta)
		# s_inx = fast_window_map_dpp(r_hat, nor_embs, args.dpp_w_size, args.num_recommendation, args.dpp_theta)
		
		# print("time used:%s" % (time.clock() - t1))
		# pdb.set_trace()
		
		x = movie_embs[:, np.array(s_inx)]
		# x = nor_embs[:, np.array(s_inx)]
		
		matrix_v = matrix_v + np.dot(x, x.T)
		
		# r = [test_items[i] if i in test_items else 0 for i in s_inx]
		reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
		# pdb.set_trace()
		
		vector_b = vector_b + np.dot(x, reward)
		# pdb.set_trace()
		# compute precision
		inter_set = set(s_inx).intersection(set(list(test_items.keys())))
		prec_curr = float(len(inter_set)) / float(args.num_recommendation)
		# print(t, prec_curr)
		prec.append(prec_curr)
		
	return np.array(prec)


def c2ucb_dpp_sketched(user_emb, movie_embs, test_items, args, num=10, lamb_da=100, sketch_m=3):
	"""
		user_emb: user embedding, shape (d, 1)
		movie_embs: movie embeddings, shape (m, d)
	"""
	hidden_dim, num_movies = movie_embs.shape
	inv_matrix_v = (1.0 / lamb_da) * np.identity(hidden_dim, dtype=np.float32)
	vector_b = np.zeros(shape=hidden_dim, dtype=np.float32)
	prec = []
	
	# feature normalization
	nor_embs = movie_embs.T.copy()
	for i in range(num_movies):
		nor_embs[i, :] = nor_embs[i, :] / np.linalg.norm(nor_embs[i, :])
	
	# initialization for sketching
	# St: m * d matrix
	St = np.zeros((sketch_m + args.num_recommendation - 1, nor_embs.shape[1]))
	
	for t in range(num):
		alpha_t = f_alpha(d=hidden_dim, t=t, m=num_movies, sigma=args.sigma,
		                  lam_da=args.lam_da, s=1)
		
		theta_hat = np.dot(inv_matrix_v, vector_b)
		
		# compute the rating scores
		r_bar = np.dot(theta_hat, movie_embs)
		r_hat = np.dot(movie_embs.T, inv_matrix_v)
		r_hat = alpha_t * np.sqrt(np.sum(r_hat.T * movie_embs, axis=0)) + r_bar
		
		# get recommendation set s
		# t1 = time.clock()
		s_inx = fast_greedy_map(r_hat, nor_embs, args.num_recommendation, args.dpp_theta)
		# s_inx = fast_window_map_dpp(r_hat, nor_embs, args.dpp_w_size, args.num_recommendation, args.dpp_theta)
		
		# print("time used:%s" % (time.clock() - t1))
		# pdb.set_trace()
		
		x = movie_embs[:, np.array(s_inx)]
		# x = nor_embs[:, np.array(s_inx)]
		
		# pdb.set_trace()
		
		St[sketch_m - 1:, :] = x.T
		St, Ht = fd_sketching(St, sketch_m, args.num_recommendation, lamb_da)
		# pdb.set_trace()
		
		# print("1 time used:%s" % (time.clock() - t1))
		# t2 = time.clock()
		# inv_matrix_v = (np.identity(hidden_dim, dtype=np.float32) - np.dot(np.dot(St.T, Ht), St)) / lamb_da
		
		inv_matrix_v = (np.identity(hidden_dim, dtype=np.float32) - np.dot(St.T, Ht * St)) / lamb_da
		
		# print("2 time used:%s" % (time.clock() - t2))
		# pdb.set_trace()
		
		# r = [test_items[i] if i in test_items else 0 for i in s_inx]
		reward = np.array([1.0 if i in test_items else 0.0 for i in s_inx])
		# pdb.set_trace()
		
		vector_b = vector_b + np.dot(x, reward)
		# pdb.set_trace()
		# compute precision
		inter_set = set(s_inx).intersection(set(list(test_items.keys())))
		prec_curr = float(len(inter_set)) / float(args.num_recommendation)
		# print(t, prec_curr)
		
		prec.append(prec_curr)
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
	for theta in [0.9]:
		test_precision = np.zeros(args.num_bandit_iter)
		args.dpp_theta = theta
		t1 = time.clock()
		length = 0.
		for user in test_user_ratings.keys():
			# prec = c2ucb_dpp_sketched(user_embs[:, user], movie_embs, test_user_ratings[user], args, num=args.num_bandit_iter)
			if len(test_user_ratings[user]) > 150:
				prec = c2ucb_dpp(user_embs[:, user], movie_embs, test_user_ratings[user], args,
								num=args.num_bandit_iter)
				# prec = c2ucb(user_embs[:, user], movie_embs, test_user_ratings[user],
				# 				args, num=args.num_bandit_iter, sim=sim_mat)
				print(user)
				print(prec)
				test_precision += prec
				length += 1
		
		print("lambda:{0}, test_precision:{1}".format(args.lam_da, test_precision / length))
		print("time used:%s" % (time.clock() - t1))


