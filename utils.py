import os
import glob
import tensorflow as tf
import numpy as np
import toml
from environment import *


import networkx as nx
import matplotlib.pyplot as plt
import re


directions = {
            0 : [1,0],
            1 : [1,-1],
            2 : [0,-1],
            3 : [-1,-1],
            4 : [-1,0],
            5 : [-1,1],
            6 : [0,1],
            7 : [1,1]
        };



def get_policy(env,target,vals):
    states = env.states
    map_data = env.map_data
    
    policy = np.zeros(vals.shape)
    
    print(vals.shape)
    
    for i in range(states.shape[0]):
        state = states[i]
        #print(state)
        val_i = [];
        
        for j in directions.keys():
        
            delta = directions[j]
            delta = np.array(delta).reshape([1,2])
            delta = np.fliplr(delta)
            next = state + delta
            next = next.astype(int)
            
            if map_data[next[0,0],next[0,1]]==1:
                val_i.append(-100)
            else:
                next_key = env.state_to_key(next);
                next_idx = env.state_lookup[next_key]
                
                val_ij = vals[next_idx]
                val_i.append(val_ij)
        val_i = np.array(val_i)
        star = np.argmax(val_i)
        policy[i] = star;
        
    return policy.reshape([-1])  


def viz_policy(target,map_data,graph,states,g_component=None):
    
    num_nodes = states.shape[0]
    
    pos = np.fliplr(states)
    node_sizes = np.array([10,]*num_nodes)
    node_color = np.array(["b",]*num_nodes)

    if g_component is not None:
        node_color[np.array(g_component,dtype=int)] = "g"
    node_color[target] = "r"
        
    #plt.imshow(map_data)
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_color)
    edges = nx.draw_networkx_edges(
            graph,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=3,
            width=1
    )
    plt.show()

def quantize(angles, levels):
    quantized_angles = np.array(angles)
    
    quantized_angles /= (2*np.pi)
    quantized_angles *= levels
    
    # CHECK: Here the % operator makes the topography cyclic. Otherwise we
    # can get 0 or 8 for the same quantum (angle=0)
    
    quantized_angles = np.round(quantized_angles)%levels
    quantized_angle_idx = np.array(quantized_angles,dtype=int)
    
    quantized_angles /= levels
    quantized_angles *= 2*np.pi

    return quantized_angles, quantized_angle_idx

def quantize_angles(sines_cosines, angles):

    
    neg_idx = np.where(angles < 0)
    angles[neg_idx] += 2*np.pi

    levels = 8
    angles, angle_idx = quantize(angles,levels)
    sines_cosines[:,0] = np.sin(angles)
    sines_cosines[:,1] = np.cos(angles)
    
    return sines_cosines, angles, angle_idx


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

def get_data_dir(experiment):
    data_dir = extract_toml(experiment)['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_Q_fnms(experiment):
    return extract_toml(experiment)['Q_fnms']



def MARE(y_true, y_pred):
    """    Mean Absolute Radian Error

    Computes the mean distance in radians between parameters.
    y_true, y_pred have form: (batch_size, (sin(a), cos(a)))

    Trignometric Identities:
        sin_ab = sin(a - b) = sin(a) cos(b) - cos(a) sin(b)
        cos_ab = cos(a - b) = cos(a) cos(b) + sin(a) sin(b)
    """
    

    sin_ab = (y_true[:, 0] * y_pred[:, 1]) - (y_true[:, 1] * y_pred[:, 0])
    cos_ab = (y_true[:, 1] * y_pred[:, 1]) - (y_true[:, 0] * y_pred[:, 0])

    return tf.reduce_mean(tf.abs(tf.atan2(sin_ab, cos_ab)))
    


def generate_map(base_data, door_data, combination):
    door_temp = door_data.copy()
    door_temp[~np.isin(door_temp, combination)] = 0
    door_temp[door_temp > 0] = 1
    return base_data + door_temp
    
    
def fnm_from_combination(combination):
    return '_'.join(re.split('\W+', str(combination)))
    
    
def read_tuple_fnm(fnm):
    with open(fnm, 'r') as f:
        tuples = [tuple(map(int, i.split())) for i in f]
        return tuples


def write_tuples_to_fnm(tuples, fnm):
    with open(fnm, 'w+') as f:
        for combo in tuples:
            f.write(' '.join(str(s) for s in combo) + '\n')    
    
    
    
    
    
    

class hyperfanin_for_kernel(tf.keras.initializers.Initializer):
    def __init__(self,fanin,varin=1.0,relu=True,bias=True):
        self.fanin = fanin
        self.varin = varin
        self.relu = relu
        self.bias = bias

    def __call__(self, shape, dtype=None, **kwargs):
        hfanin,_ = shape;
        variance = (1/self.varin)*(1/self.fanin)*(1/hfanin)

        if self.relu:
            variance *= 2.0;
        if self.bias:
            variance /= 2.0;
        
        variance = np.sqrt(3*variance);
        
        return tf.random.uniform(shape, minval=-variance, maxval=variance)
        #return tf.random.normal(shape)*variance
        
    def get_config(self):  # To support serialization
        return {"fanin": self.fanin, "varin": self.varin, "relu": self.relu, "bias": self.bias}
        
        
        

class hyperfanin_for_bias(tf.keras.initializers.Initializer):
    def __init__(self,varin=1.0,relu=True):
        self.varin = varin
        self.relu = relu

    def __call__(self, shape, dtype=None, **kwargs):
        hfanin,_ = shape;
        variance = (1/2)*(1/self.varin)*(1/hfanin)
        
        if self.relu:
            variance *= 2.0;
        
        variance = np.sqrt(3*variance);
        
        return tf.random.uniform(shape, minval=-variance, maxval=variance)
        #return tf.random.normal(shape)*variance

    def get_config(self):  # To support serialization
        return {"relu": self.relu, "varin": self.varin}

    
    
    
    

