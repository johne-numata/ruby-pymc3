import pymc3 as pm
import numpy as np

# Some data
n = 5 * np.ones(4, dtype=int)
x = np.array([-.86, -.3, -.05, .73])

# Priors on unknown parameters
alpha = pm.Normal('alpha', mu=0, tau=.01)
beta = pm.Normal('beta', mu=0, tau=.01)

# Arbitrary deterministic function of parameters
@pm.deterministic
def theta(a=alpha, b=beta):
    """theta = logit^{-1}(a+b)"""
    return pm.invlogit(a + b * x)

# Binomial likelihood for data
d = pm.Binomial('d', n=n, p=theta, value=np.array([0., 1., 3., 5.]),
                  observed=True)
