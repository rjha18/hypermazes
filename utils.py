
import tensorflow as tf;
import numpy as np;

from environment import *

import pandas as pd



import re



def load_map_fn(world_fnm,base_world_fnm,Q_fnm,batch_size,classification=False,train=True,s=None,holdout=None):
	try:
		fp = open(world_fnm, 'r')
		fp = open(base_world_fnm, 'r')
		fp.close()
	except OSError:
		print("Map file cannot be opened.")
		raise OSError()
	map_data = np.array(pd.read_csv(world_fnm,header=None,delimiter=' '));
	
	load_map(map_data,base_world_fnm,Q_fnm,batch_size,classification,train,s,holdout)


def load_map(map_data,base_world_fnm,Q_fnm,batch_size,index,classification=False,train=True,s=None,holdout=None):
	try:
		fp = open(base_world_fnm, 'r')
		fp.close()
	except OSError:
		print("Map file cannot be opened.")
		raise OSError()
		
	base_map_data = np.array(pd.read_csv(base_world_fnm,header=None,delimiter=' '));

	states = np.zeros((0,2))
	
	for ih in range(map_data.shape[0]):
		for iw in range(map_data.shape[1]):
			cell = map_data[ih,iw]
			
			if cell==0:
				state = np.array([ih,iw]).reshape([1,2])
				states = np.concatenate([states,state],axis=0)
	states = states.astype(np.float32);
	M = states.shape[0]
	Q = np.load(Q_fnm)
	
	if not train:
		S = states[(np.ones(len(states)) * s).astype(np.int32)]
		G = states[np.arange(len(states))]
		grid = np.concatenate([G,S], axis=-1)
		Q = Q[s]
	else:
		idx = np.arange(states.shape[0])
		
		grid_x,grid_y = np.meshgrid(idx,idx)
		
		grid_x = grid_x.reshape([-1])
		grid_y = grid_y.reshape([-1])

		if holdout is not None:
			grid_x = grid_x[:-(len(holdout) * states.shape[0])]
			grid_y = [x for x in grid_y if x not in holdout]
		
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
			
		np.random.seed(1337)
		
		idx = np.random.permutation(grid.shape[0])
		grid = grid[idx]
		Q = Q.reshape([-1,])[idx]
	
	if not classification:
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	

