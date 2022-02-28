import os
import numpy as np
import pandas as pd
import toml
import glob
import tensorflow as tf
from utils import load_map, generate_map


def has_splits(experiment):
    data_dir = extract_toml(experiment)['data_dir']
    return len(glob.glob(data_dir+"**/*.npy")) == 6


def extract_toml(experiment):
    toml_fnm = "experiments/" + experiment + "/experiment.toml"
    return toml.load(toml_fnm)['experiment']


def get_log_dir(experiment):
    log_dir = extract_toml(experiment)['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_splits(idx, train_pct, val_pct):
    n_train_c = int(np.floor(train_pct * len(idx)))
    n_val_c = int(np.floor(val_pct * n_train_c))
    n_train_c -= n_val_c
    return idx[:n_train_c], idx[n_train_c:n_train_c+n_val_c], idx[n_train_c+n_val_c:]
    

def gen_splits(experiment, total_states=756, total_maps=164):
    toml_data = extract_toml(experiment)
    data_dir = toml_data['data_dir']
    ground_truths_path = toml_data['Q_fnms']
    goals_density = toml_data['goal_state_train_pct']
    combination_density = toml_data['combination_train_pct']
    goals_val_pct = toml_data['goal_state_val_pct']
    combination_val_pct = toml_data['combination_val_pct']

    idx = np.random.permutation(total_maps)
    ground_truths = np.array(glob.glob(ground_truths_path))
    tr_idx, v_idx, t_idx = get_splits(idx, combination_density, combination_val_pct)
    tr_combinations = ground_truths[tr_idx]
    v_combinations = ground_truths[v_idx]
    t_combinations = ground_truths[t_idx]

    tr_states, v_states, t_states = get_splits(np.random.permutation(total_states),
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
        maps.append(np.stack([map_data, map_data - base_data]))
        datasets.append(load_map(map_data,Q_fnm,batch_size,map_offset+i,train_states=states))
    return tf.data.experimental.sample_from_datasets(datasets), tf.convert_to_tensor(maps)