import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
room = np.array(pd.read_csv('worlds/world2.grid',header=None,delimiter=' '))
result = Image.fromarray(room.astype(np.uint8))
ratio = 15
result = result.resize((room.shape[0]*ratio,room.shape[1]*ratio), resample=Image.LANCZOS)
ascent = np.array(result)
ax1.imshow(ascent, vmin=0, vmax=1)
ax2.imshow(result, vmin=0, vmax=1)
plt.savefig('abc.png')

