from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from email.mime import base
import numpy as np
from environment import *
import matplotlib.pyplot as plt
import itertools
from utils import generate_map, fnm_from_combination
import time
procedure_start = time.time()

TOTAL_DOORS = 12
NUM_DOORS = 3
base_fnm = "./worlds/3x3/gen/base.grid"
door_fnm = "./worlds/3x3/gen/doors.grid"

doors = np.arange(1, TOTAL_DOORS + 1)

base_data = np.array(pd.read_csv(base_fnm,header=None,delimiter=' '));
door_data = np.array(pd.read_csv(door_fnm,header=None,delimiter=' '));


success_times = []
failure_times = []

for combination in list(itertools.combinations(doors, NUM_DOORS)):
    start = time.time()
    map_data = generate_map(base_data, door_data, combination)
    try:
        env = gridworld_env(map_data,display=False);
        V = env.generate_V()
        
        '''
        plt.imshow(V)
        plt.colorbar()
        plt.show()
        map_states = env.states
        for idx in range(len(map_states)):
            i = idx + 450
            world = np.zeros((31,31))-1
            
            for j in range(len(map_states)):
                state = map_states[j]
                x = state[0].astype(int)
                y = state[1].astype(int)
                world[x,y] = 10*V[i,j]
            plt.subplot(1,2,1)
            plt.imshow(world)
            plt.colorbar()
            V = np.transpose(V)
            world = np.zeros((31,31))-1
            
            for j in range(len(map_states)):
                state = map_states[j]
                x = state[0].astype(int)
                y = state[1].astype(int)
                world[x,y] = 10*V[i,j]
            plt.subplot(1,2,2)
            plt.imshow(world)
            plt.colorbar()
            plt.show()
        '''
        np.save('V/3x3/' + fnm_from_combination(combination) + '.npy', V)
        total_time = time.time() - start
        success_times.append(total_time)
        print(f"Generated combination {combination} in {total_time} seconds")
    except OSError:
        total_time = time.time() - start
        failure_times.append(total_time)
        print(f"Failed combination {combination} in {total_time} seconds")
        continue

all_times = success_times + failure_times
print()
print(f"Total Procedure Time: {(time.time() - procedure_start) / 60} min")
print(f"Overall Computation Times (n = {len(all_times)}). Total: {np.sum(all_times) / 60} min, Average: {np.mean(all_times)} sec")
if len(failure_times) != 0:
    print(f"\tSuccess Times (n = {len(success_times)}): Total: {np.sum(success_times) / 60} min, Average: {np.mean(success_times)} sec")
    print(f"\tFailure Times (n = {len(failure_times)}): Total: {np.sum(failure_times) / 60} min, Average: {np.mean(failure_times)} sec")
