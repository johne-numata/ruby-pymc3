import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc3 as pm

plt.figure(figsize=(8.5, 4.5))

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1)
    data_generator = pm.Poisson("data_generator", parameter)
    data_plus_one = data_generator + 1

print(parameter.tag.test_value)

with pm.Model() as model:
    theta = pm.Exponential("theta", 2)
    data_generator = pm.Poisson("data_generator", theta)

print(theta.tag.test_value)

with pm.Model() as ab_testing:
    p_A = pm.Uniform("P(A)", 0, 1)
    p_B = pm.Uniform("P(B)", 0, 1)

print(theta.random)

print("parameter.tag.test_value =", parameter.tag.test_value)
print("data_generator.tag.test_value =", data_generator.tag.test_value)
print("data_plus_one.tag.test_value =", data_plus_one.tag.test_value)

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1, testval=0.5)

print("\nparameter.tag.test_value =", parameter.tag.test_value)

with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1)
    lambda_2 = pm.Exponential("lambda_2", 1)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)

new_deterministic_variable = lambda_1 + lambda_2

import theano.tensor as tt

with pm.Model() as theano_test:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = tt.stack([p1, p2])
    
    assignment = pm.Categorical("assignment", p)

samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")
plt.xlim(0, 8)
plt.show()
