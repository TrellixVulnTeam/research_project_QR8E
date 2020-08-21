import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn import model_selection, metrics, preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
names=np.arange(0,90,1).tolist()
#load the data
#读取文件中的内容。注意和通常读取数据的区别之处
#df=open('/Users/weihe/Desktop/triplet_angle_with_labels.pickle','rb')#注意此处是rb
#此处使用的是load(目标文件)
#data3=pickle.load(df)
data3=np.load('/Users/weihe/Desktop/triplet_angle_with_labels.pickle',allow_pickle=True)
#data=data3[:,0]
#target=data3[:,1]

size=np.array(data3).shape
#print(size)
#print(np.array(data3[0,0]).shape)
#average the 3sim into 1
for i in range(size[0]):
    data3[i,0]=data3[i][0].mean(0)

#print(np.array(data3[0,0]).shape)
#build up training set from 50
data=data3[:,0]
target=data3[:,1]
data=np.stack( data, axis=0 )

#80% for train
#20% for test
trainfrac=1
testfrac=1-trainfrac
trainsize=int(trainfrac*size[0])
testsize=int(testfrac*size[0])

#build up train set
traind=data[0:trainsize,:]
traint=target[0:trainsize]

print(traind.shape)
print(traint.shape)

x=traind
tmax=np.max(traind)
x=traind/tmax
#x=x.reshape(-1,1)

#preprocessing normaliztion
#max_abs_scaler = preprocessing.MaxAbsScaler()
#x = max_abs_scaler.fit_transform(x)
#x=preprocessing.normalize(x, norm='l2')
#x=traind[:,23]
#x=x.reshape(-1,1)
y=traint

rf=SVR(kernel='linear',C=1e7,epsilon=8)
#rf=LinearSVR(epsilon=3,loss='epsilon_insensitive'
#             ,C=1e10)
rf.fit(x, traint)
#selx=x[:,23].reshape(-1,1)
"""
z=rf.fit(x, traint).predict(selx)
plt.plot(selx, z, 'bo', label='RBF model')
plt.plot(selx,y,'ro')
plt.ylabel('y')
plt.xlabel('featrue')
plt.figure(1)
plt.show()
"""

k=rf.coef_**2
print(k)
themax=np.max(k)
p=np.where(k==themax)
print('max: ',themax)
print('loaction of max: ',p)

rfe = RFE(rf, n_features_to_select=1)
rfe.fit(x,traint)
x=x[:,50]
x=x.reshape(-1,1)
#rf=LinearSVR(epsilon=5,loss='squared_epsilon_insensitive',
#             C=100,max_iter=9999)
#rf = LinearSVR(epsilon=5,loss='squared_epsilon_insensitive'
#               ,C=50,max_iter=2000)


z=rf.fit(x, traint).predict(x)
plt.plot(x, z, 'bo', label='RBF model')
plt.plot(x,y,'ro')
plt.ylabel('y')
plt.xlabel('featrue')
plt.figure(1)
plt.show()


print("Features sorted by their rank(Ridge):")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))