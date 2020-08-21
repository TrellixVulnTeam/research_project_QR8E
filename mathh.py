import matplotlib.pyplot as plt
import numpy as np
#a is mean
#a = np.array([10, 1])
#b is std
#b = np.array([1, 10])
import st as st

mu, sigma = 0, .1
s = np.random.normal(loc=mu, scale=sigma, size=100)
print(np.std(s))