import glob, os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from tensorflow import keras
import tensorflow as tf
import argparse

import pandas as pd

from model import rlf
from utils import load_map, MARE, generate_map



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
base_world_fnm = './worlds/3x3/gen/base.grid'
door_fnm = './worlds/3x3/gen/doors.grid'

batch_size = 128;
num_maps = 50
total_maps = 164
density = 0.5

directions=['_top', '_bottom', '_left', '']

# holdout = np.random.randint(0, 726, 144)
# np.save("holdout.npy", holdout)

base_data = np.array(pd.read_csv(base_world_fnm,header=None,delimiter=' '));
door_data = np.array(pd.read_csv(door_fnm,header=None,delimiter=' '));
datasets = []
maps = []
combinations = []
idx = np.random.choice(total_maps, num_maps)
for i, Q_fnm in enumerate(np.array(glob.glob("./Q/3x3/*.npy"))[idx]):
	combination = tuple(map(int, Q_fnm[9:-5].split('_')))
	combinations.append(combination)
	map_data = generate_map(base_data, door_data, combination)
	maps.append(np.stack([map_data, map_data - base_data]))
	datasets.append(load_map(map_data,Q_fnm,batch_size,i,density,classification=CLASSIFICATION))
maps = tf.convert_to_tensor(maps)
print(combinations)
dataset = tf.data.experimental.sample_from_datasets(datasets)
# print(dataset)
# input()

writer = tf.summary.create_file_writer('./logs/{}'.format(OUTDIR))

if CLASSIFICATION:
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	f_sz[-1] = 8
else:
	loss_fn = tf.keras.losses.MeanSquaredError()
	# loss_fn = MARE

model = rlf(e_sz,f_sz,batch_size,maps,lr=1e-4,writer=writer,classification=CLASSIFICATION)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(OUTDIR), update_freq=1)
]



ds_iter = iter(dataset)
BATCH = next(ds_iter)




model.build([(batch_size),(batch_size, 4),(batch_size, 2)])
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
