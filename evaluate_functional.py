
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
parser.add_argument("--classification", help="use classification loss",default=False, action='store_true')
args = parser.parse_args()
INDIR = args.indir
TARGET = args.target
CLASSIFICATION = args.classification

	

e_sz = [64,64,16]
f_sz = [64,64,1]

world_fnm = './worlds/world8.grid'
Q_fnm = 'Q.npy'


batch_size = 332;
dataset = load_map(world_fnm,Q_fnm,batch_size,classification=CLASSIFICATION)


if CLASSIFICATION:
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	f_sz[-1] = 8
else:
	loss_fn = tf.keras.losses.MeanSquaredError()

model = rlf(e_sz,f_sz,world_fnm,batch_size,classification=CLASSIFICATION)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(INDIR), update_freq=1)
]

model.build((batch_size,4))
print(model.summary())

model.load_weights('./logs/{}/model/weights'.format(INDIR)).expect_partial()


model.compile(
	optimizer=keras.optimizers.Adam(1e-4),
	loss=loss_fn,
	metrics=[loss_fn] # add run_eagerly=True
)

# for TARGET in range(332):
dataset = load_map(world_fnm,Q_fnm,batch_size,CLASSIFICATION,False,TARGET)
results = model.predict(dataset)
if CLASSIFICATION:
	results = np.argmax(results, axis=1)
else:
	results = np.round(results)


env = gridworld_env('./worlds/world8.grid',step_penalty=0.05,gamma=0.9,display=False);
env.plot_Q(np.load(Q_fnm), TARGET, 'img/Q.png', random=True)
env.plot_results(results, TARGET, 'img/results.png', random=True)

