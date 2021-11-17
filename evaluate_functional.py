
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


from sklearn.manifold import TSNE

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
	metrics=[loss_fn], run_eagerly=True
)

# for TARGET in range(332):
dataset = load_map(world_fnm,Q_fnm,batch_size,CLASSIFICATION,False,TARGET)
results = model.predict(dataset)






states = model.states

LEFT = np.where(states[:,1]<=10)[0]
RIGHT = np.where(states[:,1]>10)[0]
UP = np.where(states[:,0]<=10)[0]
DOWN = np.where(states[:,0]>10)[0]




X = model.embedding.numpy()


X = TSNE(n_components=2,perplexity=50.0,n_iter=10000,init='random').fit_transform(X)


R1 = X[np.intersect1d(LEFT,UP),:]
R2 = X[np.intersect1d(RIGHT,UP),:]
R3 = X[np.intersect1d(RIGHT,DOWN),:]
R4 = X[np.intersect1d(LEFT,DOWN),:]


plt.scatter(R1[:,0],R1[:,1],color='r')
plt.scatter(R2[:,0],R2[:,1],color='g')
plt.scatter(R3[:,0],R3[:,1],color='y')
plt.scatter(R4[:,0],R4[:,1],color='b')
plt.show()



if CLASSIFICATION:
	results = np.argmax(results, axis=1)
else:
	results = np.round(results)


env = gridworld_env('./worlds/world8.grid',step_penalty=0.05,gamma=0.9,display=False);
env.plot_Q(np.load(Q_fnm), TARGET, 'img/Q.png', random=True)
env.plot_results(results, TARGET, 'img/results.png', random=True)

