from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
k=[0,0,1,2.5,3,5,6,7.5,8.3,9.4,10.3]
t=[0,1,2,3,4,5,6,7,8,9,10]
res=[None for _ in range(len(k))]
print(res)
print(sqrt(mean_squared_error(k, t)))

#rf=LinearSVR(C=1000, epsilon=2, max_iter=10000000.0, tol=0.01)
i=[25.110345127556993, 13.39517808502038, 9.770904068292355, 28.30468993910201, 12.162024821663168]
#rf=LinearSVR(C=1000, epsilon=0.35, max_iter=10000000.0, tol=0.1)
j=[23.01874,8.06943283,14.11711,22.83156548,14.672522]

avgi=np.average(i)
avgj=np.average(j)
print(avgi)
print(avgj)

#print(np.average[10.2504572,13.9670287,6.857501,18.56338749,6.03876276])


