require 'pycall/import'
include PyCall::Import

pyimport 'pymc3', as: :pm
pyimport 'theano.tensor', as: :tt

#with pm.Model() as model:
#    alpha = 1.0/count_data.mean()  # Recall count_data is the
#                                   # variable that holds our txt counts
#    lambda_1 = pm.Exponential("lambda_1", alpha)
#    lambda_2 = pm.Exponential("lambda_2", alpha)
#    
#    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)