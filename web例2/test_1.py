
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
returns= pd.read_csv("https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/SP500.csv",
                     header=-1, parse_dates=True)[2500:2900]

#plt.style.use('ggplot')
#returns.columns =['S&P500']
#returns.plot(figsize=(12,7), c="b")
#plt.show()

from pymc3 import Exponential, T, exp, Deterministic, Model, sample, NUTS, find_MAP, traceplot
from pymc3.distributions.timeseries import GaussianRandomWalk

with Model() as sp500_model:
    nu = Exponential('nu', 1./10, testval=5.)
    sigma = Exponential('sigma', 1./.02, testval=.1)
    s = GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = Deterministic('volatility_process', exp(-2*s))
    r = T('r', nu, lam=1/volatility_process, observed=returns['S&P500'])



