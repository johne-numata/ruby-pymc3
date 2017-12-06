import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc as pm

count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
alpha = 1.0 / count_data.mean()

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)
lambda_3 = pm.Exponential("lambda_3", alpha)

tau_1 = pm.DiscreteUniform("tau_1", lower=0, upper= n_count_data - 1)
tau_2 = pm.DiscreteUniform("tau_2", lower=0, upper= n_count_data - 1)

@pm.deterministic
def lambda_(tau_1=tau_1, tau_2=tau_2, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3):
	out = np.zeros(n_count_data)
	out[:tau_1] = lambda_1
	out[tau_1:tau_2] = lambda_2
	out[tau_2:] = lambda_3
	return out

observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)
model = pm.Model([observation, lambda_1, lambda_2, lambda_3, tau_1, tau_2])
mcmc = pm.MCMC(model)
mcmc.sample(100000, 30000)

lambda_1_samples = mcmc.trace("lambda_1")[:]
lambda_2_samples = mcmc.trace("lambda_2")[:]
lambda_3_samples = mcmc.trace("lambda_3")[:]
tau1_samples = mcmc.trace('tau_1')[:]
tau2_samples = mcmc.trace('tau_2')[:]

ax = plt.subplot(511)
ax.set_autoscaley_on(False)
plt.hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
	color="#A60628", normed=True, label="posterior of $\lambda_1$")
plt.legend(loc="upper left")
plt.title("Posterior distributions of the five unknown prameters in the extend text-message model")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")
plt.ylabel("Density")

ax = plt.subplot(512)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
	color="#7A68a6", normed=True, label="posterior of $\lambda_2$")
plt.legend(loc="upper left")
plt.xlim([30, 90])
plt.xlabel("$\lambda_2$ value")
plt.ylabel("Density")

ax = plt.subplot(513)
ax.set_autoscaley_on(False)
plt.hist(lambda_3_samples, histtype="stepfilled", bins=30, alpha=0.85,
	color="#7A68A6", normed=True, label="posterior of $\lambda_3$")
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")
plt.ylabel("Density")

plt.subplot(514)
w = 1.0 / tau1_samples.shape[0] * np.ones_like(tau1_samples)
plt.hist(tau1_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau_1$",
	color="#467821", weights=w)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0, 0.75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel("Dey")
plt.ylabel("Probability")

plt.subplot(515)
w = 1.0 / tau2_samples.shape[0] * np.ones_like(tau2_samples)
plt.hist(tau2_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau_2$",
	color="#467821", weights=w)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0, 0.75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel("Dey")
plt.ylabel("Probability")

print(lambda_1_samples.mean())
print(lambda_2_samples.mean())
print(lambda_3_samples.mean())
print(tau1_samples.mean())
print(tau2_samples.mean())

plt.show()
