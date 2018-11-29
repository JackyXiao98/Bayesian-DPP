import os
import numpy as np
import argparse
import scipy.io as sio
import csv

from sklearn.cluster import KMeans
import timeit

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score

import os
import sys
import time
import socket
import numpy as np
import scipy.ndimage as ndi
import inspect
if sys.version_info[0]==3:
    import _pickle as cPickle
else:
    import cPickle


################################# ramp up & ramp down #################################

def ramp_up(x_max=1.0, x_min=0.0, t=None, t_max=600, t_start=0, t_stop=None, type='linear', exp_hp=-5.0):
    if t_stop == None:
        t_stop = t_max
    t = float(t)
    x_max = float(x_max)
    if (t_stop<t_start) or (t_stop>t_max) :
        raise ValueError("t_start < t_stop < t_max is not valid!")
    if x_min > x_max:
        raise ValueError("x_min > x_max  is not valid!")

    if t < t_start:
        x = x_min

    elif (t <= t_stop) and (t >= t_start):
        if type == 'exp':
            p = 1.0 - (t-t_start) / (t_stop-t_start)
            x = x_max * np.exp(exp_hp*p*p) + x_min
        elif type == 'linear':
            x = ((x_max-x_min) * t + (x_min*t_stop - x_max*t_start)) / (t_stop-t_start)
        elif type == 'cte':
            x = x_min
        else:
            raise ValueError("type is not defined")

    elif (t > t_stop):
        x = x_max

    return np.asarray(x, dtype=np.float32)


def ramp_down(x_max=1.0, x_min=0.0, t=None, t_max=600, t_start=0, t_stop=None, type='linear', exp_hp=-5.0):
    if t_stop == None:
        t_stop = t_max
    t = float(t)
    x_max = float(x_max)
    if (t_stop<t_start) or (t_stop>t_max) :
        raise ValueError("t_start < t_stop < t_max is not valid!")
    if x_min > x_max:
        raise ValueError("x_min > x_max  is not valid!")

    if t < t_start:
        x = x_max

    elif (t <= t_stop) and (t >= t_start):
        if type == 'exp':
            p = (t - t_start) / (t_stop - t_start)
            x = (x_max-x_min) * np.exp(exp_hp * p * p) + x_min
        elif type == 'linear':
            x = ((x_min - x_max) * t + (x_max * t_stop - x_min * t_start)) / (t_stop - t_start)
        elif type == 'cte':
            x = x_max
        else:
            raise ValueError("type is not defined")

    elif (t > t_stop):
        x = x_min

    return np.asarray(x, dtype=np.float32)


################################# Logging #################################
class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_path, "log.txt"), "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def create_result_dirs(output_path, file_name, myfunc_list=[], verbose=0):
    if not os.path.exists(output_path):
        if verbose > 0:  print('\n*** Creating logging folder ***')
        os.makedirs(output_path)


        myfunc_list += ['data.py', 'tf_net_new.py']
        for func in myfunc_list:
            func_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), func)
            cmd = 'cp ' + func_full_path + ' "' + output_path + '"'
            os.popen(cmd)

        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)

def logging(file_name=None, cuda=None, verbose=0, test_err=None, output_path=None, myfunc_list=[]):
    if output_path == None:
        output_path = './results/' + file_name.split('.')[0] + '/' + time.strftime("%m_%d_%H_%M_%S")
        create_result_dirs(output_path, file_name, myfunc_list=myfunc_list, verbose=verbose)
        sys.stdout = Logger(output_path)
        if verbose > 1: print(sys.argv)
    else:
        os.rename(output_path, output_path + '_' + str(test_err))

    return output_path
#
# ################################# Logging #################################
# def create_result_dirs(file_name, output_path):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#         func_file_name = os.path.basename(__file__)
#         if func_file_name.split('.')[1] == 'pyc':
#             func_file_name = func_file_name[:-1]
#         functions_full_path = os.path.join(output_path, func_file_name)
#         cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
#         os.popen(cmd)
#         run_file_full_path = os.path.join(output_path, file_name)
#         cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
#         os.popen(cmd)
#
# class Logger(object):
#     def __init__(self, file_name=None, output_path=None, verbose=0):
#         self.file_name = file_name
#         self.verbose = verbose
#         if self.verbose > 0:  print('\n*** Creating logging folder ***')
#         self.output_path = output_path
#         if self.output_path == None:
#             output_path = './results/' + file_name.split('.')[0] + '/' + time.strftime("%d-%m-%Y_") +\
#                           time.strftime("%H:%M:%S_")  + socket.gethostname()
#         create_result_dirs(file_name, output_path)
#
#         self.terminal = sys.stdout
#         self.log = open(os.path.join(output_path, "log.txt"), "w+")
#
#
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         # this flush method is needed for python 3 compatibility.
#         # this handles the flush command by doing nothing.
#         # you might want to specify some extra behavior here.
#         pass
#
#     def rename(self, test_err):
#             os.rename(self.output_path, self.output_path + '_' + format(test_err))
#



################################# print results #################################

def print_results(ep=None, t=None,
                  l_tot=None, l_tot_std=None, l_ent=None, l_rec=None, l_cons=None, l_reg=None, l_disc=None, l_gen=None,
                  l_orth=None, l_coll=None, l_prior=None, l_wd=None, l_trace=None, l_enc=None, l_mae=None,
                  v_sl=None,
                  gan_acc=None, precision=None, mAP=None, mAP_k=None, best_flag=None,
                  acc=None, acc_std=None, nmi=None, nmi_std=None, arc=None, arc_std=None,
                  mae=None, ae_std=None, mcc=None, cc_std=None, mae_va=None, ae_std_va=None, mcc_va=None, cc_std_va=None,
                  tr_err=None, va_err=None, te_err=None, va_err2=None, te_err2=None, b_va_err=None, b_va_err2=None):


    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)


    prt_str = ""
    for i in args:
        if values[i] is not None:
            if (i == 'end') or (i == 'flush'):
                continue
            elif (i == 't') or (i == 'ep'):
                str_value = str(int(values[i]))
            elif i[:2] == 'l_':
                str_value = format(values[i], '.3f')
            elif i[3:] == '_acc':
                str_value = format(values[i], '.2f')
            elif i[2:6] == '_err':
                str_value = format(values[i], '.4f')
            else:
                str_value = format(values[i], '.4f')

            prt_str += str(i) + ' = ' + str_value + ', '

    print(prt_str)