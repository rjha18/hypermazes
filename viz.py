import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.imshow(np.array(pd.read_csv('worlds/world8_bottom.grid',header=None,delimiter=' ')), cmap='Greys')
plt.savefig('img/world8.png')