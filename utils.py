import os
import glob
import tensorflow as tf
import numpy as np
import toml
from environment import *


import re

def has_splits(experiment):
    data_dir = extract_toml(experiment)['data_dir']
    return len(glob.glob(data_dir+"**/*.npy")) == 6


def extract_toml(experiment):
    toml_fnm = "experiments/" + experiment + "/experiment.toml"
    return toml.load(toml_fnm)['experiment']


def get_log_dir(experiment):
    log_dir = extract_toml(experiment)['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_data_dir(experiment):
    data_dir = extract_toml(experiment)['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_Q_fnms(experiment):
    return extract_toml(experiment)['Q_fnms']



def MARE(y_true, y_pred):
	"""	Mean Absolute Radian Error

	Computes the mean distance in radians between parameters.
	y_true, y_pred have form: (batch_size, (sin(a), cos(a)))

	Trignometric Identities:
		sin_ab = sin(a - b) = sin(a) cos(b) - cos(a) sin(b)
		cos_ab = cos(a - b) = cos(a) cos(b) + sin(a) sin(b)
	"""
	

	sin_ab = (y_true[:, 0] * y_pred[:, 1]) - (y_true[:, 1] * y_pred[:, 0])
	cos_ab = (y_true[:, 1] * y_pred[:, 1]) - (y_true[:, 0] * y_pred[:, 0])

	return tf.reduce_mean(tf.abs(tf.atan2(sin_ab, cos_ab)))
	


def generate_map(base_data, door_data, combination):
	door_temp = door_data.copy()
	door_temp[~np.isin(door_temp, combination)] = 0
	door_temp[door_temp > 0] = 1
	return base_data + door_temp
	
	
def fnm_from_combination(combination):
    return '_'.join(re.split('\W+', str(combination)))
	
	
def read_tuple_fnm(fnm):
	with open(fnm, 'r') as f:
		tuples = [tuple(map(int, i.split())) for i in f]
		return tuples


def write_tuples_to_fnm(tuples, fnm):
	with open(fnm, 'w+') as f:
		for combo in tuples:
			f.write(' '.join(str(s) for s in combo) + '\n')	
	
	
	
	
	
	
	
	
	
	

