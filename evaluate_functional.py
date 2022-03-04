
import os


import numpy as np

from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf
from utils import get_Q_fnms, fnm_from_combination, get_log_dir, get_data_dir, quantize_angles, extract_toml, setup_model
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




if COMBINATION==None and COMB_SPLIT==None:
    print('Either an evaluation combination or an evaluation combination split must be provided.')
    exit(-1)
    
if COMBINATION==None:
    eval_combinations = list(np.load(data_dir + COMB_SPLIT + '/combinations.npy'))
    idx = np.random.choice(len(eval_combinations))
    COMBINATION = tuple(map(int, eval_combinations[idx][9:-5].split('_')))
    print('Choosing random eval combination from '+COMB_SPLIT+' split:', COMBINATION)
else:
    combo_fnm = get_Q_fnms(EXPERIMENT).replace('*', fnm_from_combination(COMBINATION))

    if combo_fnm in list(np.load(data_dir + 'train/combinations.npy')):
        print("Combination provided belongs to train split.")
    elif combo_fnm in list(np.load(data_dir + 'val/combinations.npy')):
        print("Combination provided belongs to validation split.")
    elif combo_fnm in list(np.load(data_dir + 'test/combinations.npy')):
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


batch_size = 756

dataset, maps, Q, map_data = generate_dataset_from_target(EXPERIMENT, TARGET, COMBINATION)

model = setup_model(EXPERIMENT,batch_size,maps);
model.evaluate(dataset)

sines_cosines = model.forward_pass(next(iter(dataset))).numpy()
angles = np.arctan2(sines_cosines[:, 0], sines_cosines[:, 1])

sines_cosines, angles, _ = quantize_angles(sines_cosines, angles)
    
    

Q = Q[TARGET] * np.pi / 4.0
Q_raw = np.stack([np.sin(Q), np.cos(Q)], axis=1)
Q = np.expand_dims(Q, axis=1)

env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);
env.plot_results(Q, Q_raw, TARGET, 'img/Q.png', random=True, target_num=7, wall_num=-2)
env.plot_results(angles, sines_cosines, TARGET, 'img/results.png', random=True, target_num=7, wall_num=-2)


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
