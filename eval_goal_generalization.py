
import os


import numpy as np

from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf
from utils import get_Q_fnms, fnm_from_combination, get_log_dir, get_data_dir, quantize_angles, directions, extract_toml
from environment import gridworld_env
from data_utils import generate_dataset_from_target


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="Name of experiment to run", type=str)
parser.add_argument('--combination', help="combination to examine", nargs=3, default=None, type=int)
parser.add_argument('--comb_split', help="combination split to sample", default=None, type=str)
parser.add_argument('--visualize', help="visualzie random policy", default=0, type=int)

args = parser.parse_args()
EXPERIMENT = args.experiment
if args.combination is not None:
    COMBINATION = tuple(args.combination)
else:
    COMBINATION = args.combination
COMB_SPLIT = args.comb_split
VISUALIZE = args.visualize



log_dir = get_log_dir(EXPERIMENT)
data_dir = get_data_dir(EXPERIMENT)


e_sz = [64,64,16]
f_sz = [32,32,1]
batch_size = 756



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
       

# map_data does not change in this experiment
_, maps, _, map_data = generate_dataset_from_target(EXPERIMENT, 0, COMBINATION)
env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);



loss_fn = tf.keras.losses.MeanSquaredError()

toml_data = extract_toml(EXPERIMENT)
model = rlf(e_sz,f_sz,batch_size,maps,method=toml_data['hypernet'],lr=1e-4)
callbacks = [keras.callbacks.TensorBoard(log_dir, update_freq=1)]

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=[loss_fn], run_eagerly=True
)

model.build([(batch_size),(batch_size, 4),(batch_size, 2)])
print(model.summary())

model.load_weights(log_dir + 'model/weights').expect_partial()



import networkx as nx
import matplotlib.pyplot as plt

train_targets = list(np.load(data_dir + 'train/states.npy'))
test_targets = list(np.load(data_dir + 'test/states.npy'))


N_tr = len(train_targets)
N_ts = len(train_targets)

accs = []


if VISUALIZE:
    random_target = np.random.randint(300,600)

for TARGET in test_targets:
    dataset, maps, Q, map_data = generate_dataset_from_target(EXPERIMENT, TARGET, COMBINATION)

    sines_cosines = model.forward_pass(next(iter(dataset))).numpy()
    angles = np.arctan2(sines_cosines[:, 0], sines_cosines[:, 1])
   
    
    sines_cosines, angles, angle_idx = quantize_angles(sines_cosines, angles)
    env.plot_results(angles, sines_cosines, TARGET, target_num=7, wall_num=-2,display_arrows=True)
    A = np.zeros((756,756))
    
    for u in range(756):
        u_xy = env.states[u]
        
        delta = directions[angle_idx[u]]
        delta = np.array(delta).reshape([1,2])
        delta = np.fliplr(delta)

        v_xy = u_xy + delta
        v_key = env.state_to_key(v_xy)
        
        if v_key not in env.state_lookup.keys():
            continue;
            
        v = env.state_lookup[v_key]
        
        A[u,v] = 1
        
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    
    
    components = nx.weakly_connected_components(G)
    
    for component in sorted(components, key=len, reverse=True):
        if TARGET in component:
            g_component = list(component)
            break;
    


    if VISUALIZE:
        if TARGET==random_target:
            pos = np.fliplr(env.states)
            node_sizes = np.array([3,]*756)
            node_color = np.array(["b",]*756)
            node_color[TARGET] = "r"
            
            node_color[np.array(g_component,dtype=int)] = "g"
            
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
            edges = nx.draw_networkx_edges(
                G,
                pos,
                node_size=node_sizes,
                arrowstyle="->",
                arrowsize=3,
                width=1
            )
            plt.show()
    
    V_ts = len(g_component)
    
    target_acc = V_ts/756
    accs.append(target_acc)    
print(np.mean(accs))
input()
                

neg_idx = np.where(angles < 0)
angles[neg_idx] += 2*np.pi

if QUANTIZE:
    levels = 8
    angles = quantize(angles,levels)
    sines_cosines[:,0] = np.sin(angles)
    sines_cosines[:,1] = np.cos(angles)
    
    

Q = Q[TARGET] * np.pi / 4.0
Q_raw = np.stack([np.sin(Q), np.cos(Q)], axis=1)
Q = np.expand_dims(Q, axis=1)

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
