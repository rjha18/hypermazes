import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.imshow(np.array(pd.read_csv('worlds/3x3/gen/doors.grid',header=None,delimiter=' ')), cmap='Greys')
plt.savefig('img/world8_base.png')