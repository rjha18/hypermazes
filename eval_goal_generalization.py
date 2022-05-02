
import os


import numpy as np

from tensorflow import keras
import tensorflow as tf
import argparse
import time
import networkx as nx

from model import rlf, setup_model
from utils import get_Q_fnms, fnm_from_combination, get_log_dir, get_data_dir, quantize_angles, directions, extract_toml, viz_policy, get_policy
from environment import gridworld_env
from data_utils import generate_dataset_from_target,generate_train_val,generate_test,inspect_settings

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="Name of experiment to run", type=str)
parser.add_argument('--combination', help="combination to examine", nargs=3, default=None, type=int)
parser.add_argument('--comb_split', help="combination split to sample", default=None, type=str)
parser.add_argument('--visualize', help="visualize policy", default=0, type=int)
parser.add_argument('--target_split', help="state split to sample", default='test', type=str)
parser.add_argument('--max_test_combos', help="maximum combinations used for testing", default=20, type=int)
parser.add_argument('--plain', help="use plain accuracy instead of reachability", default=0, type=int)

args = parser.parse_args()
EXPERIMENT = args.experiment
if args.combination is not None:
    COMBINATION = tuple(args.combination)
else:
    COMBINATION = args.combination
COMB_SPLIT = args.comb_split
VISUALIZE = args.visualize
TARGET_SPLIT = args.target_split
PLAIN = args.plain

MAX_COMBOS = args.max_test_combos

ALL = COMBINATION==None and COMB_SPLIT==None
    


log_dir = get_log_dir(EXPERIMENT)
data_dir = get_data_dir(EXPERIMENT)





    
    
if COMBINATION==None:
    
    if COMB_SPLIT==None:
        eval_combinations = list(np.load(data_dir + 'test/combinations.npy'))
        eval_combinations = np.array(eval_combinations)
        
        # change this for deterministic testing
        #eval_combinations = np.random.permutation(eval_combinations)[:MAX_COMBOS]
        eval_combinations = eval_combinations[:MAX_COMBOS]
        
        COMBINATIONS = [tuple(map(int, eval_combination[9:-5].split('_'))) for eval_combination in eval_combinations]
        print('Evaluating ' +str(MAX_COMBOS) + ' test combinations!')
    else:
        eval_combinations = list(np.load(data_dir + COMB_SPLIT + '/combinations.npy'))
        idx = np.random.choice(len(eval_combinations))
        COMBINATIONS = [tuple(map(int, eval_combinations[idx][9:-5].split('_')))]
        print('Choosing random eval combination from '+COMB_SPLIT+' split:', COMBINATION)
else:
    COMBINATIONS = COMBINATION
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
        
        

'''
_, val_dataset, maps = generate_train_val(EXPERIMENT, batch_size)
model = setup_model(EXPERIMENT,batch_size,maps,load=True,eager=True);
print(model.evaluate(val_dataset))

test_dataset, maps = generate_test(EXPERIMENT, batch_size)
model = setup_model(EXPERIMENT,batch_size,maps,load=True,eager=True);
print(model.evaluate(test_dataset))

'''

counter = 0
targets = list(np.load(data_dir + TARGET_SPLIT + '/states.npy'))
    
num_nodes = 756
batch_size = num_nodes
    
    
accs = []
Z = []
    
#train_dataset, val_dataset, maps = generate_train_val(EXPERIMENT, batch_size)
#model = setup_model(EXPERIMENT,batch_size,maps,load=True,eager=True);
#model.evaluate(train_dataset)
#model.evaluate(val_dataset)



for combo in COMBINATIONS:
    _, maps, _, map_data = generate_dataset_from_target(EXPERIMENT, 0, combo)
    env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);

    t = time.time()
    
    #if counter==0:
    model = setup_model(EXPERIMENT,batch_size,maps,load=True,eager=True);
    #else:
    #    model.maps = maps
    elapsed = time.time() - t
        
    print(counter)
    counter += 1

    N = len(targets)



    if VISUALIZE:
        targets = np.random.permutation(targets)

    
    for TARGET in targets:
        dataset, maps, Q, map_data = generate_dataset_from_target(EXPERIMENT, TARGET, combo)

        if PLAIN:
            target_acc = model.evaluate(dataset,verbose=0)
        else:
            BATCH = next(iter(dataset))

            sines_cosines = model.forward_pass(next(iter(dataset)),training=False).numpy()
                
            angles = np.arctan2(sines_cosines[:, 0], sines_cosines[:, 1])
            ANGLES = np.arange(8)*np.pi/4
            
            
            
            angle_idx = np.argmax(sines_cosines,axis=-1)
            angles = ANGLES[angle_idx]
            
            
            sines_cosines = np.zeros((batch_size,2))
            sines_cosines[:,0] = np.sin(angles)
            sines_cosines[:,1] = np.cos(angles)
            #sines_cosines, angles, angle_idx = quantize_angles(sines_cosines, angles)
        
            #print(angles.shape)
            
            
            
            #Z += [model.z[0]]
            
            
            A = np.zeros((756,756))
            
            for u in range(756):

                if u==TARGET:
                    continue
                if angle_idx[u]==8:
                    continue
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
                
            At = np.transpose(A)
            G = nx.from_numpy_matrix(At, create_using=nx.DiGraph)
            
            paths = dict(nx.all_pairs_shortest_path_length(G))
            
            g_component = list(paths[TARGET].keys())
            V_ts = len(np.intersect1d(g_component,targets))
            # Alternative using weakly connected components
            G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            components = nx.weakly_connected_components(G)
            
            for component in sorted(components, key=len, reverse=True):
                if TARGET in component:
                    g_component = list(component)
                    break;
            V_ts = len(np.intersect1d(g_component,targets))
            
            
            target_acc = V_ts/N
            
            
            #print(len(targets))
            #print(V_ts)
            #print(len(g_component))
            #print('Spot Accuracy: ',target_acc)
            
            if VISUALIZE:
                env.plot_results(angles, sines_cosines, TARGET, random=True, target_num=7, wall_num=-2)
                plt.show()
                G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
                viz_policy(TARGET,targets,map_data,G,env.states,g_component=g_component)
                #break;
                
        accs.append(target_acc)    
    print('Accuracy: ',np.mean(accs))
'''
Z = np.array(Z,dtype=float)
print(Z)
from sklearn.manifold import TSNE
X = TSNE(n_components=2).fit_transform(Z)
print(X.shape)


plt.scatter(X[:,0],X[:,1])
plt.show()

exit(0)   

if not VISUALIZE:
    print('Accuracy: ',np.mean(accs))
       
'''
       
       
       
       
       
       
       
       
       
       
       
