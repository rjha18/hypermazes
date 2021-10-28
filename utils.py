
import tensorflow as tf;
import numpy as np;

from environment import *

import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras

from PIL import Image
from scipy.io import loadmat







def load_map(world_fnm,Q_fnm,batch_size):

	try:
		fp = open(world_fnm, 'r')
		fp.close()
	except OSError:
		print("Map file cannot be opened.")
		raise OSError()
		
	map_data = np.array(pd.read_csv(world_fnm,header=None,delimiter=' '));

	states = np.zeros((0,2))
	
	for ih in range(map_data.shape[0]):
		for iw in range(map_data.shape[1]):
			cell = map_data[ih,iw]
			
			if cell==0:
				state = np.array([ih,iw]).reshape([1,2])
				states = np.concatenate([states,state],axis=0)
	states = states.astype(np.float32);
	
	M = states.shape[0]
	
	Q = np.load(Q_fnm).reshape([-1,])

	
	idx = np.arange(states.shape[0])
	
	grid_y,grid_x = np.meshgrid(idx,idx)
	
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
		
	
	np.random.seed(1337)
	
	idx = np.random.permutation(grid.shape[0])
	grid = grid[idx]
	Q = Q[idx]*np.pi/4.0
	Q = Q.astype(np.float32)
	
	
	tr_ds = tf.data.Dataset.from_tensor_slices((grid,Q)).batch(batch_size,drop_remainder=False)
	
	return tr_ds;
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

