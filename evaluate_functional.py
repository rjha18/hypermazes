
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf
from utils import get_Q_fnms, fnm_from_combination, get_log_dir, get_data_dir
from environment import gridworld_env
from data_utils import generate_dataset_from_target


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="Name of experiment to run", type=str)
parser.add_argument("--target", help="the id of the target index to evaluate",default=-1, type=int)
parser.add_argument('--combination', help="combination to examine", nargs=3, default=[-1, -1, -1], type=int)

args = parser.parse_args()
EXPERIMENT = args.experiment
TARGET = args.target
COMBINATION = tuple(args.combination)


log_dir = get_log_dir(EXPERIMENT)
data_dir = get_data_dir(EXPERIMENT)


e_sz = [64,64,16]
f_sz = [64,64,2]
batch_size = 756


test_combinations = list(np.load(data_dir + 'test/combinations.npy'))
combo_fnm = get_Q_fnms(EXPERIMENT).replace('*', fnm_from_combination(COMBINATION))
if sum(COMBINATION) < 0:
	idx = np.random.choice(len(test_combinations))
	COMBINATION = tuple(map(int, test_combinations[idx][9:-5].split('_')))
	print(COMBINATION)
elif combo_fnm not in test_combinations:
	print("WARNING: the combination provided is in the train set!")
	input()

test_states = list(np.load(data_dir + 'test/states.npy'))
if TARGET < 0:
	TARGET = np.random.choice(test_states)
elif TARGET not in test_states:
	print("WARNING: the target provided is in the train set!")
	input()

dataset, maps, Q, map_data = generate_dataset_from_target(EXPERIMENT, TARGET, COMBINATION)


loss_fn = tf.keras.losses.MeanSquaredError()
model = rlf(e_sz, f_sz, batch_size, maps, lr=1e-4)
callbacks = [keras.callbacks.TensorBoard(log_dir, update_freq=1)]


model.build([(batch_size),(batch_size, 4),(batch_size, 2)])
print(model.summary())

model.load_weights(log_dir + 'model/weights').expect_partial()


model.compile(
	optimizer=keras.optimizers.Adam(1e-4),
	loss=loss_fn,
	metrics=[loss_fn], run_eagerly=True
)


results = model.forward_pass(next(iter(dataset)))
results_parsed = np.arctan2(results[:, 0], results[:, 1])
idx = np.where(results_parsed < 0)
results_parsed[idx] = results_parsed[idx] + 2 * np.pi

Q = Q[TARGET] * np.pi / 4.0
Q_raw = np.stack([np.sin(Q), np.cos(Q)], axis=1)
Q = np.expand_dims(Q, axis=1)

env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);
env.plot_results(Q, Q_raw, TARGET, 'img/Q.png', random=True, target_num=7, wall_num=-2)
env.plot_results(results_parsed, results, TARGET, 'img/results.png', random=True, target_num=7, wall_num=-2)


''' WEIGHTS viz code
states = model.states

LEFT = np.where(states[:,1]<=10)[0]
RIGHT = np.where(states[:,1]>10)[0]
UP = np.where(states[:,0]<=10)[0]
DOWN = np.where(states[:,0]>10)[0]


from sklearn.decomposition import PCA

X = model.embedding.numpy()
pca = PCA(n_components=2)
X = pca.fit_transform(X)


R1 = X[np.intersect1d(LEFT,UP),:]
R2 = X[np.intersect1d(RIGHT,UP),:]
R3 = X[np.intersect1d(RIGHT,DOWN),:]
R4 = X[np.intersect1d(LEFT,DOWN),:]

plt.scatter(R1[:,0],R1[:,1],color='r')
plt.scatter(R2[:,0],R2[:,1],color='g')
plt.scatter(R3[:,0],R3[:,1],color='y')
plt.scatter(R4[:,0],R4[:,1],color='b')
plt.savefig('img/weights.png')
'''