import numpy as np

import matplotlib.pyplot as plt


x = np.linspace(0, 3e-6, 10000)
distr = 1 / (1 + np.exp((x - 2.3e-6) / 5e-8))

plt.plot(x, distr)
plt.show()
