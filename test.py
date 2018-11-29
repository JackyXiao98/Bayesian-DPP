import numpy as np

def broad_cast():
	a = np.array([[1, 2, 3], [2, 3, 4]])
	b = a.T
	a2 = a[:, None, :]
	a3 = a[:, :, None]
	aha = a2*a3
	b2 = b[:, None, :]
	b3 = b[:, :, None]
	bhb = b2*b3
	sum_of_bhb = np.sum(bhb, axis=0)
	sum_of_aha = np.sum(aha, axis=0)
	
	vector_m = np.array([2,2,2])
	matrix_m = np.tile(vector_m,(2,1))
	
	print("hi")


def rho(x):
	# if x > 0:
	# 	return 1. / (1. + np.exp(-x))
	# else:
	# 	return np.exp(x) / (np.exp(x) + 1.)
	return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))


x = np.array([1, -1])
x_wh = np.where(x > 0, 1, -1)

print("ok")


