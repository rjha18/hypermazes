
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import argparse

import importlib

from model import rlf
from utils import load_map
from environment import gridworld_env


parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--target", help="the id of the target index to evaluate",default=0, type=int)
args = parser.parse_args()
INDIR = args.indir
TARGET = args.target

	

e_sz = [64,64,16]
f_sz = [64,64,1]

world_fnm = './worlds/world8.grid'
Q_fnm = 'Q.npy'


batch_size = 332;
dataset = load_map(world_fnm,Q_fnm,batch_size)




model = rlf(e_sz,f_sz,world_fnm,batch_size)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(INDIR), update_freq=1)
]

model.build((batch_size,4))
print(model.summary())

model.load_weights('./logs/{}/model/weights'.format(INDIR)).expect_partial()


model.compile(
	optimizer=keras.optimizers.Adam(1e-4),
	loss=tf.keras.losses.MeanSquaredError(),
	metrics=[tf.keras.losses.MeanSquaredError()] # add run_eagerly=True
)


dataset = load_map(world_fnm,Q_fnm,batch_size,False,TARGET)
results = model.predict(dataset)


#model.forward(something)
# E_S = model.embedding.numpy()


env = gridworld_env('./worlds/world8.grid',step_penalty=0.05,gamma=0.9,display=False);
env.plot_Q(np.load(Q_fnm), TARGET, 'Q.png', random=True)
env.plot_results(results, TARGET, 'results.png', random=True)

