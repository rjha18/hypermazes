import numpy as np
from environment import *
import matplotlib.pyplot as plt


batch_size = 32;
GAMMA = 0.9;


grid_lvl = 8;
env = gridworld_env('./worlds/world'+str(grid_lvl)+'.grid',step_penalty=0.05,gamma=GAMMA,display=False);

# env.plot_vals('abc.png', random=True)
Q = env.generate_Q()
print("animation")
input()
for i in range(332):
    env.plot_Q(Q, i, 'abc.png', random=True)
    plt.clf()