import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import pandas as pd

from environment import gridworld_env
from utils import generate_map

base_fnm = "./worlds/3x3/gen/base.grid"
door_fnm = "./worlds/3x3/gen/doors.grid"

Q = np.load('Q/3x3/0.npy')
wall_num = -1
target_num = 8

Q = Q * np.pi / 4.0
target_num = 7
wall_num = -2

base_data = np.array(pd.read_csv(base_fnm,header=None,delimiter=' '));
door_data = np.array(pd.read_csv(door_fnm,header=None,delimiter=' '));

map_data = generate_map(base_data, door_data, (1, 2, 3))
env = gridworld_env(map_data,step_penalty=0.05,gamma=0.9,display=False);
env.plot_Q(Q, 0, 'img/Q2.png', random=True, target_num=target_num, wall_num=wall_num)