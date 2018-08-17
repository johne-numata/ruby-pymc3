import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc3 as pm
from mpl_toolkits.mplot3d import Axes3D

jet = plt.cm.jet
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12.5, 5))

plt.subplot(121)
exp_x = stats.expon.pdf(x, scale=3)
exp_y = stats.expon.pdf(x, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])
CS = plt.contour(X, Y, M)
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
#plt.xlabel("prior on $p_1$")
#plt.ylabel("prior on $p_2$")
plt.title("$Exp(3), Exp(10)$ prior landscape")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=jet)
ax.view_init(azim=390)
plt.title("$Exp(3), Exp(10)$ prior landscape; \nalternate view");
plt.show()
