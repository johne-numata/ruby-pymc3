import pymc3
import mymodel

S = pymc3.MCMC(mymodel, db='pickle')
S.sample(iter=10000, burn=5000, thin=2)
pymc3.Matplot.plot(S)

