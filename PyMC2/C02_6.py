import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc as pm

plt.figure(figsize=(12.5, 10))

N = 100
p = pm.Uniform("freq_cheating", 0, 1)

@pm.deterministic
def p_skewed(p=p):
    return 0.5 * p + 0.25

yes_responses = pm.Binomial("number_cheaters", 100, p_skewed, value=35, observed=True)
model = pm.Model([yes_responses, p_skewed, p])

# To Be Explained in Chapter 3!
mcmc = pm.MCMC(model)
mcmc.sample(25000, 2500)

p_trace = mcmc.trace("freq_cheating")[:]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()
plt.show()
