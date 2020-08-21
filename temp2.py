import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn import model_selection, metrics
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge

#load the data
#读取文件中的内容。注意和通常读取数据的区别之处
#df=open('/Users/weihe/Desktop/triplet_angle_with_labels.pickle','rb')#注意此处是rb
#此处使用的是load(目标文件)
#data3=pickle.load(df)
data3=np.load('/Users/weihe/Desktop/triplet_angle_with_labels.pickle',allow_pickle=True)
#data=data3[:,0]
#target=data3[:,1]

size=np.array(data3).shape
print(size)
print(np.array(data3[0,0]).shape)
#average the 3sim into 1
for i in range(size[0]):
    data3[i,0]=data3[i][0].mean(0)

print(np.array(data3[0,0]).shape)
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
names=np.arange(0,90,1).tolist()
# use linear regression as the model
#lr = LinearSVR(epsilon=5,loss='squared_epsilon_insensitive'
#               ,C=100,max_iter=2000)
lr=LinearSVR()
lr2=Ridge()
lr3=Lasso()
lr4=LinearRegression()
#CANNOT USE lr5=SVR(kernel='poly',epsilon=2)
#same as above lr5=KernelRidge(kernel='poly')

# rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe2 = RFE(lr2, n_features_to_select=1)
rfe3 = RFE(lr3, n_features_to_select=1)
rfe4 = RFE(lr4,n_features_to_select=1)

rfe.fit(traind, traint)
rfe2.fit(traind,traint)
rfe3.fit(traind,traint)
rfe4.fit(traind,traint)


print("Features sorted by their rank(LinearSVR):")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

print("Features sorted by their rank(Ridge):")
print(sorted(zip(map(lambda x: round(x, 4), rfe2.ranking_), names)))
print("Features sorted by their rank(Lasso):")
print(sorted(zip(map(lambda x: round(x, 4), rfe3.ranking_), names)))
print("Features sorted by their rank(linearregression):")
print(sorted(zip(map(lambda x: round(x, 4), rfe4.ranking_), names)))
print("Features sorted by their rank(SVR):")
print(sorted(zip(map(lambda x: round(x, 4), rfe4.ranking_), names)))
#print(rfe.feature_importances_)
#print(sorted(zip(map(lambda x: round(x, 4), rfe.feature_importances_), names),reverse=True))
x=traind[:,22]
x=x.reshape(-1,1)
z=lr.fit(x, traint).predict(x)
plt.plot(x, z, 'bo', label='RBF model')
plt.plot(x,traint,'ro')
plt.ylabel('y')
plt.xlabel('featrue')
plt.figure(1)
plt.show()