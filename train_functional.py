
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import argparse

import importlib

from model import rlf






parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--load", help="load model",default=0, type=int)
parser.add_argument("--epochs", help="# of training epochs",default=100, type=int)
args = parser.parse_args()
OUTDIR = args.outdir
LOAD = args.load
EPOCHS = args.epochs

	

e_sz = [64,64,16]
f_sz = [64,64,8]

world_fnm = './worlds/world8.grid'

model = rlf(e_sz,f_sz,world_fnm)

Q = np.load('Q.npy')
num_states = Q.shape[0]

callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(OUTDIR), update_freq=1)
]

model.build([model.map_data.shape,(num_states*num_states,)])
print(model.summary())

if LOAD:
	model.load_weights('./logs/{}/model/weights'.format(OUTDIR)).expect_partial()
	
	

model.compile(
	optimizer=keras.optimizers.Adam(1e-3),
	loss=tf.keras.losses.MeanSquaredError(),
	metrics=[tf.keras.losses.MeanSquaredError()]
)


model.fit(
	tr_ds,
	callbacks = callbacks,
	epochs = EPOCHS,
	verbose = 1
)

model.save_weights('./logs/{}/model/weights'.format(OUTDIR))


