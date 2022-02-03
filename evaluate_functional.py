
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf
from utils import load_map, MARE, generate_map, fnm_from_combination
from environment import gridworld_env



parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="Directory for results and log files", default='./log', type=str)
parser.add_argument("--target", help="the id of the target index to evaluate",default=0, type=int)
parser.add_argument("--direction", help="direction of wall",default="top", type=str)
parser.add_argument("--classification", help="use classification loss",default=False, action='store_true')
args = parser.parse_args()
INDIR = args.indir
TARGET = args.target
DIRECTION = args.direction
CLASSIFICATION = args.classification
combination = (1, 2, 3)

	

e_sz = [64,64,16]
f_sz = [64,64,2]

base_world_fnm = './worlds/3x3/gen/base.grid'
door_fnm = './worlds/3x3/gen/doors.grid'
Q_fnm = './Q/3x3/' + fnm_from_combination(combination) + '.npy'
base_data = np.array(pd.read_csv(base_world_fnm,header=None,delimiter=' '));
door_data = np.array(pd.read_csv(door_fnm,header=None,delimiter=' '));
map_data = generate_map(base_data, door_data, combination)

maps = tf.convert_to_tensor([np.stack([map_data, map_data - base_data])])


batch_size = 756;

if CLASSIFICATION:
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	f_sz[-1] = 8
else:
	loss_fn = tf.keras.losses.MeanSquaredError()

model = rlf(e_sz,f_sz,batch_size,maps,lr=1e-4,classification=CLASSIFICATION)


callbacks = [
	keras.callbacks.TensorBoard(
		'./logs/{}'.format(INDIR), update_freq=1)
]


model.build([(batch_size),(batch_size, 4),(batch_size, 2)])
print(model.summary())

model.load_weights('./logs/{}/model/weights'.format(INDIR)).expect_partial()


model.compile(
	optimizer=keras.optimizers.Adam(1e-4),
	loss=loss_fn,
	metrics=[loss_fn], run_eagerly=True
)

# for TARGET in range(332):

dataset = load_map(map_data,base_world_fnm,Q_fnm,batch_size,0,CLASSIFICATION,False,TARGET)
ds_iter = iter(dataset)
BATCH = next(ds_iter)
results = model.forward_pass(BATCH)


states = model.states

LEFT = np.where(states[:,1]<=10)[0]
RIGHT = np.where(states[:,1]>10)[0]
UP = np.where(states[:,0]<=10)[0]
DOWN = np.where(states[:,0]>10)[0]


from sklearn.decomposition import PCA

X = model.embedding.numpy()
pca = PCA(n_components=2)
X = pca.fit_transform(X)

#X = TSNE(n_components=2,n_iter=10000,init='random').fit_transform(X)

R1 = X[np.intersect1d(LEFT,UP),:]
R2 = X[np.intersect1d(RIGHT,UP),:]
R3 = X[np.intersect1d(RIGHT,DOWN),:]
R4 = X[np.intersect1d(LEFT,DOWN),:]

plt.scatter(R1[:,0],R1[:,1],color='r')
plt.scatter(R2[:,0],R2[:,1],color='g')
plt.scatter(R3[:,0],R3[:,1],color='y')
plt.scatter(R4[:,0],R4[:,1],color='b')
plt.savefig('img/weights.png')

np.save(DIRECTION + ".npy", model.hyperembedding.numpy())


Q = np.load(Q_fnm)
wall_num = -1
target_num = 8
if CLASSIFICATION:
	results = np.argmax(results, axis=1)
else:
	Q = Q * np.pi / 4.0
	target_num = 7
	wall_num = -2
	results = np.arctan2(results[:, 0], results[:, 1])
	idx = np.where(results < 0)
	results[idx] = results[idx] + 2 * np.pi

print("Results", results.shape)
env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);
env.plot_Q(Q, TARGET, 'img/Q.png', random=True, target_num=target_num, wall_num=wall_num)
env.plot_results(results, TARGET, 'img/results.png', random=True, target_num=target_num, wall_num=wall_num)

