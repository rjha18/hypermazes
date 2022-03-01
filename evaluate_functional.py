
import os


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
parser.add_argument("--target", help="the id of the target index to evaluate",default=None, type=int)
parser.add_argument('--combination', help="combination to examine", nargs=3, default=None, type=int)
parser.add_argument('--comb_split', help="combination split to sample", default=None, type=str)
parser.add_argument('--target_split', help="state split to sample", default=None, type=str)

args = parser.parse_args()
EXPERIMENT = args.experiment
TARGET = args.target
if args.combination is not None:
    COMBINATION = tuple(args.combination)
else:
    COMBINATION = args.combination
COMB_SPLIT = args.comb_split
TARGET_SPLIT = args.target_split



log_dir = get_log_dir(EXPERIMENT)
data_dir = get_data_dir(EXPERIMENT)


e_sz = [64,64,16]
f_sz = [32,32,1]
batch_size = 756

combo_fnm = get_Q_fnms(EXPERIMENT).replace('*', fnm_from_combination(COMBINATION))

if COMBINATION==None and COMB_SPLIT==None:
    print('Either an evaluation combination or an evaluation combination split must be provided.')
    exit(-1)
    
if COMBINATION==None:
    eval_combinations = list(np.load(data_dir + COMB_SPLIT + '/combinations.npy'))
    idx = np.random.choice(len(eval_combinations))
    COMBINATION = tuple(map(int, eval_combinations[idx][9:-5].split('_')))
    print('Choosing random eval combination from '+COMB_SPLIT+' split:', COMBINATION)
else:
    if COMBINATION in list(np.load(data_dir + 'train/combinations.npy')):
        print("Combination provided belongs to train split.")
    elif COMBINATION in list(np.load(data_dir + 'val/combinations.npy')):
        print("Combination provided belongs to validation split.")
    elif COMBINATION in list(np.load(data_dir + 'test/combinations.npy')):
        print("Combination provided belongs to test split.")
    else:
        print("Combination does not exist.")
        exit(-1)
       
if TARGET==None and TARGET_SPLIT==None:
    print('Either an evaluation target or an evaluation target split must be provided.')
    exit(-1) 

if TARGET==None:
    eval_states = list(np.load(data_dir + TARGET_SPLIT +'/states.npy'))
    TARGET = np.random.choice(eval_states)
    print('Choosing random eval state from '+TARGET_SPLIT+' split:', TARGET)
else:
    if TARGET in list(np.load(data_dir + 'train/states.npy')):
        print("Target provided belongs to train split.")
    elif TARGET in list(np.load(data_dir + 'val/states.npy')):
        print("Target provided belongs to validation split.")
    elif TARGET in list(np.load(data_dir + 'test/states.npy')):
        print("Target provided belongs to test split.")
    else:
        print("Target does not exist.")
        exit(-1)

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
