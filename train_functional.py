
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






parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--load", help="load model",default=0, type=int)
parser.add_argument("--epochs", help="# of training epochs",default=10, type=int)
parser.add_argument("--classification", help="use classification loss",default=False, action='store_true')
args = parser.parse_args()
OUTDIR = args.outdir
LOAD = args.load
EPOCHS = args.epochs
CLASSIFICATION = args.classification

	

e_sz = [64,64,16]
f_sz = [64,64,1]

world_fnm = './worlds/world8.grid'
Q_fnm = 'Q.npy'


batch_size = 32;
dataset = load_map(world_fnm,Q_fnm,batch_size,classification=CLASSIFICATION)



writer = tf.summary.create_file_writer('./logs/{}'.format(OUTDIR))

if CLASSIFICATION:
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	f_sz[-1] = 8
else:
	loss_fn = tf.keras.losses.MeanSquaredError()

model = rlf(e_sz,f_sz,world_fnm,batch_size,writer=writer,classification=CLASSIFICATION)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(OUTDIR), update_freq=1)
]

model.build((None,4))
print(model.summary())

if LOAD:
	model.load_weights('./logs/{}/model/weights'.format(OUTDIR)).expect_partial()
	
	

model.compile(
	optimizer=keras.optimizers.Adam(1e-4),
	loss=loss_fn,
	metrics=[loss_fn]
)


model.fit(
	dataset,
	callbacks = callbacks,
	epochs = EPOCHS,
	verbose = 1
)

model.save_weights('./logs/{}/model/weights'.format(OUTDIR))


