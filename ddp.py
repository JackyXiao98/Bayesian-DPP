"""
	This file includes the functions for fast DPP inference.
"""
import pdb
import numpy as np
from collections import defaultdict


def fast_greedy_map(scores, movie_embs, K, theta):
	"""
		movie_embs: m * d
	"""
	C = defaultdict(list)
	D = scores * scores
	D = D * np.sum(movie_embs * movie_embs, axis=1)
	j = np.argmax(D)
	Y = [j]
	m = scores.shape[0]
	Z = set(range(m))
	alpha = theta / (1 - theta)
	# vec = alpha * scores
	vec = 1.0 / (1.0 + np.exp(-scores))
	vec = vec**(alpha)

	for k in range(1, K):
		Z = Z - set(Y)
		for i in Z:
			# pdb.set_trace()
			Sji = (np.sum(movie_embs[j] * movie_embs[i]) + 1) / 2
			# Lji = scores[j] * scores[i] * Sji
			Lji = vec[j] * vec[i] * Sji
			# pdb.set_trace()
			if len(C[i]) == 0 or len(C[j]) == 0:
				ei = Lji / (D[j]**(0.5))
			else:
				ei = (Lji - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j]**(0.5))
			C[i].append(ei)
			D[i] = D[i] - ei**(2)
		ii = np.array(list(Z))
		jj = np.argmax(D[ii])
		j = ii[jj]
		Y.append(j)
	# pdb.set_trace()
	return Y


def fast_window_map_dpp(scores, movie_embs, w_size, N, theta):
	"""
		w_size: the size of the window
	"""
	C = defaultdict(list)
	D = scores * scores
	D = D * np.sum(movie_embs * movie_embs, axis=1)
	j = np.argmax(D)
	Y = [j]
	m = scores.shape[0]
	Z = set(range(m))
	A = defaultdict(float)

	# initialize V
	V = np.zeros((w_size, w_size))
	V[0, 0] = D[j]**(0.5)

	alpha = theta / (1 - theta)
	# vec = alpha * scores
	vec = 1.0 / (1.0 + np.exp(-scores))
	vec = vec**(alpha)

	for T in range(1, N):
		# update V

		# if T > 1:
		# 	# pdb.set_trace()
		# 	x1 = np.zeros(V.shape[0]).reshape((V.shape[0], 1))
		# 	V = np.concatenate((V, x1), axis=1)
		# 	x1 = np.array(C[j] + [D[j]**(0.5)]).reshape((1, len(C[j])+1))
		# 	V = np.concatenate((V, x1), axis=0)

		if T <= w_size:
			V[T-1, :T] = np.array(C[j] + [D[j]**(0.5)])

		else:
			V[:w_size-1, :w_size-1] = V1
			V[w_size-1, :] = np.array(C[j] + [D[j]**(0.5)])

		Z = Z - set(Y)

		for i in Z:
			# pdb.set_trace()
			Sji = (np.sum(movie_embs[j] * movie_embs[i]) + 1) / 2
			# Lji = scores[j] * scores[i] * Sji
			Lji = vec[j] * vec[i] * Sji
			# pdb.set_trace()
			if len(C[i]) == 0 or len(C[j]) == 0:
				ei = Lji / (D[j]**(0.5))
			else:
				ei = (Lji - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j]**(0.5))
			C[i].append(ei)
			D[i] = D[i] - ei**(2)

		if len(Y) >= w_size:
			v = V[1:, 0]
			V1 = V[1:, 1:]
			for i in Z:
				A[i] = C[i][0]
				C[i] = C[i][1:]

			for k in range(w_size - 1):
				tt = V1[k,k]**(2) + v[k]**(2)
				t = tt**(0.5)
				V1[k+1:, k] = (V1[k+1:, k] * V1[k, k] + v[k+1:] * v[k]) / t
				v[k+1:] = (v[k+1:] * t - V1[k+1:, k] * v[k]) / V1[k, k]

				for i in Z:
					C[i][k] = (C[i][k] * V1[k, k] + A[i] * v[k]) / t
					A[i] = (A[i] * t - C[i][k] * v[k]) / V1[k, k]
				V1[k, k] = t

			for i in Z:
				D[i] = D[i] +A[i]**(2)

		ii = np.array(list(Z))
		jj = np.argmax(D[ii])
		j = ii[jj]
		Y.append(j)

	return Y


