require 'pycall/import'
include PyCall::Import

pyimport 'pymc', as: :pm
#pyimport 'theano.tensor', as: :tt
pyimport 'numpy', as: :np
pyimport 'matplotlib.pyplot', as: :plt
pyimport 'scipy.stats', as: :stats

size = PyCall::Tuple.([12.5, 6])
plt.figure.({figsize: size})

def logistic(x, beta, alpha=0)
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))
end

np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
# drop the NA values
#challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

#temperature = challenger_data[:, 0]
#D = challenger_data[:, 1]  # defect or not?






