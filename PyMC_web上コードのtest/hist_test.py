from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(12.5, 4)

colors = ["#348ABD", "#A60628"]
prior = [1/21., 20/21.]
posterior = [0.087, 1 - 0.087]
plt.bar([0, 7], prior, alpha = 0.70, width = 0.25,
		color= colors[0], label="prior distribution",
		lw = "3", edgecolor= "#348ABD")

plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
		width = 0.25, color=colors[1],
		label = "posterior distribution",
		lw = "3", edgecolor="#A60628")

plt.xticks([0.20, 0.95], ["lobrarian", "Farmer"])
plt.ylabel("Probablity")
plt.legend(loc="upper left")
plt.title("Prior and posterior probabirities of Steve's occupation")
plt.show()
