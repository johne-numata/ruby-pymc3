import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

with pm.Model():    
    x = pm.Normal("x", mu=0, sd=1)
    trace = pm.sample(1000)

plt.plot(trace['x'])
