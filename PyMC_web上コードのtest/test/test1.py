import numpy as np 
import pymc3 as pm 

x_sample = np.random.normal(loc=1.0, scale=1.0, size=1000) 

with pm.Model() as model: 
	mu = pm.Normal('mu', mu=0., sd=0.1) 
	x = pm.Normal('x', mu=mu, sd=1., observed=x_sample) 


with model: 
	start = pm.find_MAP() 
	step = pm.NUTS() 
	trace = pm.sample(10000, step, start) 


pm.traceplot(trace).savefig("result1.jpg") 
