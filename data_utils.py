import os
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from utils import generate_map, extract_toml, fnm_from_combination



def inspect_settings(experiment):
    toml_data = extract_toml(experiment)
    data_dir = toml_data['data_dir']
    
    for split in ['train','val','test']:
        split_dir = data_dir+split
        combinations = np.load(split_dir+'/combinations.npy')
        states = np.load(split_dir+'/states.npy')

        print(combinations)
        print(states)
        input()
    

def get_splits_by_num(idx, num_train_maps, num_val_maps):
    assert num_train_maps > num_val_maps
    num_train_left = num_train_maps - num_val_maps
    return idx[:num_train_left], idx[num_train_left:num_train_maps], idx[num_train_maps:]
    
    
    
def get_splits_by_pct(idx, train_pct, val_pct):
    n_train_c = int(np.floor(train_pct * len(idx)))
    n_val_c = int(np.floor(val_pct * n_train_c))
    n_train_c -= n_val_c
    return idx[:n_train_c], idx[n_train_c:n_train_c+n_val_c], idx[n_train_c+n_val_c:]
    
    

def gen_splits(experiment, total_states=756, total_maps=164):
    toml_data = extract_toml(experiment)
    data_dir = toml_data['data_dir']
    ground_truths_path = toml_data['Q_fnms']
    goals_density = toml_data['goal_state_train_pct']
    goals_val_pct = toml_data['goal_state_val_pct']
    
    num_train_maps = toml_data['num_train_maps']
    num_val_maps = toml_data['num_val_maps']

    idx = np.random.permutation(total_maps)
    ground_truths = np.array(glob.glob(ground_truths_path))
    tr_idx, v_idx, t_idx = get_splits_by_num(idx, num_train_maps, num_val_maps)
    
    print(tr_idx)
    input()
    print(v_idx)
    input()
    print(t_idx)
    input()
    print(len(tr_idx))
    input()
    print(len(v_idx))
    input()
    print(len(t_idx))
    input()
    tr_combinations = ground_truths[tr_idx]
    v_combinations = ground_truths[v_idx]
    t_combinations = ground_truths[t_idx]

    tr_states, v_states, t_states = get_splits_by_pct(np.random.permutation(total_states),
                                               goals_density, goals_val_pct)

    os.makedirs(data_dir + 'train', exist_ok=True)
    os.makedirs(data_dir + 'val', exist_ok=True)
    os.makedirs(data_dir + 'test', exist_ok=True)

    np.save(data_dir + 'train/combinations.npy', tr_combinations)
    np.save(data_dir + 'val/combinations.npy', v_combinations)
    np.save(data_dir + 'test/combinations.npy', t_combinations)

    np.save(data_dir + 'train/states.npy', tr_states)
    np.save(data_dir + 'val/states.npy', v_states)
    np.save(data_dir + 'test/states.npy', t_states)


def generate_train_val(experiment, batch_size):
    train_dataset, maps = generate_dataset_from_splits(experiment, "train", batch_size)
    val_dataset, val_maps = generate_dataset_from_splits(experiment, "val", batch_size, maps.shape[0])
    maps = tf.concat([maps, val_maps], axis=0)
    return train_dataset, val_dataset, maps


def generate_dataset_from_splits(experiment, split, batch_size, map_offset=0):
    experiment_data = extract_toml(experiment)
    data_dir = experiment_data['data_dir']
    base_data = np.array(pd.read_csv(experiment_data['base_fnm'], header=None, delimiter=' '));
    door_data = np.array(pd.read_csv(experiment_data['doors_fnm'], header=None, delimiter=' '));
    
    combinations = np.load(data_dir + split + "/combinations.npy")
    states = np.load(data_dir + split + "/states.npy")

    datasets = []
    maps = []
    for i, Q_fnm in enumerate(combinations):
        combination = tuple(map(int, Q_fnm[9:-5].split('_')))
        map_data = generate_map(base_data, door_data, combination)
        map_states = generate_map_states(map_data)
        maps.append(np.stack([map_data, map_data - base_data]))
        Q = np.load(Q_fnm)
        datasets.append(create_batch_dataset(map_states, Q, batch_size, map_offset + i, states))
    return tf.data.experimental.sample_from_datasets(datasets), tf.convert_to_tensor(maps)


def generate_dataset_from_target(experiment, target, combination):
    experiment_data = extract_toml(experiment)
    base_data = np.array(pd.read_csv(experiment_data['base_fnm'], header=None, delimiter=' '));
    door_data = np.array(pd.read_csv(experiment_data['doors_fnm'], header=None, delimiter=' '));

    map_data = generate_map(base_data, door_data, combination)
    map_states = generate_map_states(map_data)
    
    Q_fnm = experiment_data['Q_fnms'].replace('*', fnm_from_combination(combination))
    Q = np.load(Q_fnm)

    maps = tf.convert_to_tensor([np.stack([map_data, map_data - base_data])])
    dataset = create_target_dataset(map_states,Q,target)
    return dataset, maps, Q, map_data


def generate_map_states(map_data):
    states = np.zeros((0,2))
    
    for ih in range(map_data.shape[0]):
        for iw in range(map_data.shape[1]):
            cell = map_data[ih,iw]
            
            if cell==0:
                state = np.array([ih,iw]).reshape([1,2])
                states = np.concatenate([states,state],axis=0)
    return states.astype(np.float32)


def compute_Q_sin_cos(Q):
    Q_new = Q*np.pi/4.0
    Q_new = np.stack((np.sin(Q_new), np.cos(Q_new)), axis=1)
    Q_new = Q_new.astype(np.float32)
    return Q_new


def create_target_dataset(map_states,Q,target):
    S = map_states[np.arange(len(map_states))]
    G = map_states[(np.ones(len(map_states)) * target).astype(np.int32)]
    grid = np.concatenate([S,G], axis=-1)
    Q = Q[target]
    
    Q = compute_Q_sin_cos(Q)
    
    maps = np.ones(grid.shape[0]) * 0

    return tf.data.Dataset.from_tensor_slices((maps, grid, Q)).batch(len(map_states))


def create_batch_dataset(map_states,Q,batch_size,index,indices):
    idx_s = np.arange(map_states.shape[0])
    
    grid_x, grid_y = np.meshgrid(idx_s, indices)
    grid_x = grid_x.reshape([-1])
    grid_y = grid_y.reshape([-1])
    
    S = map_states[grid_x]
    G = map_states[grid_y]
    grid = np.concatenate([S,G],axis=-1)

    idx = np.random.permutation(grid.shape[0])
    grid = grid[idx]
    Q = Q[indices].reshape([-1,])[idx]
    
    Q = compute_Q_sin_cos(Q)
    maps = np.ones(grid.shape[0]) * index

    return tf.data.Dataset.from_tensor_slices((maps, grid, Q)).batch(batch_size,drop_remainder=True)
