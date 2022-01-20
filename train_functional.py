
import os
import numpy as np
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import argparse

import importlib

from model import rlf
from utils import load_map, MARE



parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--load", help="load model",default=0, type=int)
parser.add_argument("--epochs", help="# of training epochs",default=50, type=int)
parser.add_argument("--classification", help="use classification loss",default=False, action='store_true')
args = parser.parse_args()
OUTDIR = args.outdir
LOAD = args.load
EPOCHS = args.epochs
CLASSIFICATION = args.classification

	

e_sz = [64,64,16]
f_sz = [64,64,2]


batch_size = 32;

directions=['_top', '_bottom', '_left']
datasets = []
for direction in directions:
	world_fnm = './worlds/world8'+direction+'.grid'
	Q_fnm = 'Q/Q'+direction+'.npy'
	datasets.append(load_map(world_fnm,Q_fnm,batch_size,classification=CLASSIFICATION))

dataset = tf.data.experimental.sample_from_datasets(datasets)

writer = tf.summary.create_file_writer('./logs/{}'.format(OUTDIR))

if CLASSIFICATION:
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	f_sz[-1] = 8
else:
	loss_fn = tf.keras.losses.MeanSquaredError()
	# loss_fn = MARE

model = rlf(e_sz,f_sz,world_fnm,batch_size,lr=1e-4,writer=writer,classification=CLASSIFICATION)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(OUTDIR), update_freq=1)
]



ds_iter = iter(dataset)
BATCH = next(ds_iter)




model.build([(batch_size, 21, 21),(batch_size, 4),(batch_size, 2)])
print(model.summary())

if LOAD:
	model.load_weights('./logs/{}/model/weights'.format(OUTDIR)).expect_partial()
	
	

model.compile(
	optimizer=keras.optimizers.Adam(1e-5),
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


