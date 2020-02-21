import numpy as np

# show the bias in sample variance

# vars for IQ
pop_mean = 100.0
pop_sigma = 15.0
pop_size = 1000000

population = np.random.normal(pop_mean, pop_sigma, pop_size)