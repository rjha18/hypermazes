import glob, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

from tensorflow import keras
import tensorflow as tf
import argparse

import pandas as pd

from model import rlf
from utils import load_map, MARE, generate_map, read_tuple_fnm, write_tuples_to_fnm

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--load_model", help="load model", default=False, action='store_true')
parser.add_argument("--load_train", help="load train set", default=False, action='store_true')
parser.add_argument("--load_holdout", help="load holdout set", default=False, action='store_true')
parser.add_argument("--epochs", help="# of training epochs", default=50, type=int)
parser.add_argument("--n_maps", help="# of maps to train with", default=50, type=int)
parser.add_argument("--classification", help="use classification loss", default=False,)

args = parser.parse_args()
OUTDIR = args.outdir
LOAD_MODEL = args.load_model
LOAD_TRAIN = args.load_train
LOAD_HOLDOUT = args.load_holdout
EPOCHS = args.epochs
N_MAPS = args.n_maps
CLASSIFICATION = args.classification


e_sz = [64,64,16]
f_sz = [64,64,2]
base_world_fnm = './worlds/3x3/gen/base.grid'
door_fnm = './worlds/3x3/gen/doors.grid'

NUM_DOORS = 3
batch_size = 128;
num_maps = 75
total_maps = 164
density = 0.25


combinations_fnm = './data/combinations.txt'
holdout_fnm = './data/holdout.txt'
ground_truths = "./Q/3x3/*.npy"


base_data = np.array(pd.read_csv(base_world_fnm,header=None,delimiter=' '));
door_data = np.array(pd.read_csv(door_fnm,header=None,delimiter=' '));
datasets = []
maps = []

combinations = []
if LOAD_TRAIN:
	combinations = read_tuple_fnm(combinations_fnm)

if LOAD_HOLDOUT:
	holdout = np.genfromtxt(holdout_fnm, dtype=int)
else:
	num_states = base_data.size - np.count_nonzero(base_data.flatten()) - (NUM_DOORS * 3)
	holdout = np.random.choice(num_states, int(np.floor((1 - density) * num_states)), replace=False)
	with open(holdout_fnm, 'w+') as f:
		f.write('\n'.join(holdout.astype(str)))

idx = np.random.choice(total_maps, num_maps, replace=False)
for i, Q_fnm in enumerate(np.array(glob.glob(ground_truths))[idx]):
	combination = tuple(map(int, Q_fnm[9:-5].split('_')))
	if not LOAD_TRAIN:
		combinations.append(combination)
	map_data = generate_map(base_data, door_data, combination)
	maps.append(np.stack([map_data, map_data - base_data]))
	datasets.append(load_map(map_data,Q_fnm,batch_size,i,classification=CLASSIFICATION,holdout=holdout))
maps = tf.convert_to_tensor(maps)

if not LOAD_TRAIN:
	write_tuples_to_fnm(combinations, combinations_fnm)


dataset = tf.data.experimental.sample_from_datasets(datasets)


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

if LOAD_MODEL:
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
