
import tensorflow as tf;
import numpy as np;

from environment import *

import pandas as pd



import re



def load_map(map_data,Q_fnm,batch_size,index,train=True,s=None,train_states=None):
	states = np.zeros((0,2))
	
	for ih in range(map_data.shape[0]):
		for iw in range(map_data.shape[1]):
			cell = map_data[ih,iw]
			
			if cell==0:
				state = np.array([ih,iw]).reshape([1,2])
				states = np.concatenate([states,state],axis=0)
	states = states.astype(np.float32);
	Q = np.load(Q_fnm)
	
	if not train:
		S = states[np.arange(len(states))]
		G = states[(np.ones(len(states)) * s).astype(np.int32)]
		grid = np.concatenate([S,G], axis=-1)
		Q = Q[s]
	else:
		idx_s = np.arange(states.shape[0])
		
		grid_x,grid_y = np.meshgrid(idx_s,train_states)
		
		grid_x = grid_x.reshape([-1])
		grid_y = grid_y.reshape([-1])
		
		S = states[grid_x]
		G = states[grid_y]
		grid = np.concatenate([S,G],axis=-1)
		
		'''
		print(grid)
		input()
		
		for i in range(grid.shape[0]):
			print(grid[i])
			input()
		'''
		idx = np.random.permutation(grid.shape[0])
		grid = grid[idx]
		Q = Q[train_states].reshape([-1,])[idx]
	
	Q = Q*np.pi/4.0
	Q = np.stack((np.sin(Q), np.cos(Q)), axis=1)
	
	Q = Q.astype(np.float32)
	
	maps = np.ones(grid.shape[0]) * index
	
	# print("Maps shape:", maps.shape)
	# print("Grid shape:", grid.shape)
	# print("Q shape:", Q.shape)
	# print("Maps GB:", maps.nbytes / 1e9)
	# print("Grid GB:", grid.nbytes / 1e9)
	# print("Q GB:", Q.nbytes / 1e9)
	# print("Total Estimate GB (before change):", ((maps.nbytes + grid.nbytes + Q.nbytes) / 1e9) * 164)
	# print("Total Estimate GB (after change):", ((grid.nbytes + Q.nbytes + (Q.nbytes / 2) + maps[0].nbytes) / 1e9) * 164)

	# input()
	tr_ds = tf.data.Dataset.from_tensor_slices((maps, grid, Q)).batch(batch_size,drop_remainder=train)
	# tr_ds = tf.data.Dataset.from_tensor_slices((grid, Q)).batch(batch_size,drop_remainder=train)
	
	return tr_ds;
	

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
	
	
	
	
	
	
	
	
	
	

