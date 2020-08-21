import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, GridSearchCV, KFold
from sklearn import model_selection, metrics, preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import time

names=np.arange(0,90,1).tolist()
#load the data
#读取文件中的内容。注意和通常读取数据的区别之处
#df=open('/Users/weihe/Desktop/triplet_angle_with_labels.pickle','rb')#注意此处是rb
#此处使用的是load(目标文件)
#data3=pickle.load(df)
data3=np.load("C:/Users/xmanf/OneDrive/Desktop/triplet_angle_with_labels.pickle",allow_pickle=True)
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
'''
#manual separate
#80% for train
#20% for test
trainfrac=1
testfrac=1-trainfrac
trainsize=int(trainfrac*size[0])
testsize=int(testfrac*size[0])

#build up train set
traind=data[0:trainsize,:]
traint=target[0:trainsize]
'''
x_temp=data;
y=target;

'''
pro_min_t=traind[traint == np.min(traint)].reshape(-1,1)
pro_max_t=traind[traint == np.max(traint)].reshape(-1,1)
print(sum(pro_max_t))
'''
"""
plt.plot(names,pro_min_t)
plt.plot(names,pro_max_t)
plt.show()
"""
#normalized
#min-max normalization for feature
tmax=np.max(x_temp)
tmin=np.min(x_temp)
x=(x_temp-tmin)/(tmax-tmin)
x=x[:,:]

##x=x_temp
#for i in range(0,58):
#    plt.plot(names,x[i,:])
#plt.show()

#for test
#train rfsvr
#rf=SVR(kernel='linear',C=1e3,epsilon=10,max_iter=-1)
#use linear svr---need cv matrix
#need to choose variable
rf=LinearSVR(C=1000, epsilon=0.2, max_iter=10000000.0, tol=0.1)
x=x[:,0:90]
#print(x.shape)

rfe = RFE(rf,n_features_to_select=1)
#set up k-fold
kf = KFold(n_splits=5, shuffle=False)
#create array for split
splitarr = np.arange(x.shape[0])

def cvtrain(splitarr):
    rmse = []
    for train_index , test_index in kf.split(splitarr):
        #set up train and test set
        traind=x[train_index,:]
        traint=y[train_index]
        testd=x[test_index,:]
        testt=y[test_index]
        #train and get rmse
        rfe.fit(traind, traint)
        rmse.append(sqrt(mean_squared_error(testt, rfe.predict(testd))))
    return np.average(rmse)

print(cvtrain(splitarr))
#print("rmse:",rmse)
#print("sum rmse:",np.sum(rmse))

'''
#self-written grid search for parameter
c=[1000]
e=[2]
t=[0.01]
bestpara=[1e5 for _ in range(4)]
for i in range(len(c)):
    for j in range(len(e)):
        for k in range(len(t)):
            start = time.process_time()
            avg_rmse = 0
            for q in range(2):
                temp_rmse=cvtrain(splitarr)
                avg_rmse=temp_rmse+avg_rmse
            avg_rmse=avg_rmse/2
            if bestpara[3]>avg_rmse:
                bestpara[0]=i
                bestpara[1]=j
                bestpara[2]=k
                bestpara[3]=avg_rmse
            print('current para',bestpara)
            print("current position:i:%d, j:%d, k:%d, rmse:%f" %(i,j,k,avg_rmse))
            end = time.process_time()
            print('time used:', end - start)
            print()

print(bestpara)
'''

#[2, 5, 0, 10.394313897241357]
#[1e5,0.5,0.01]

#[2, 5, 0, 11.95281341367831]
#[1e6,1,0.001]

#[1, 0, 2, 11.938663912704971]
#[1e6,0.5,0.01]

#[0, 0, 2, 13.464482668109639]
#[1e4,0.4,0.01]

#c=[1e3,2000,5000,8000,1e4,20000,50000,1e5,1e6]
#e=[0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,1]
#t=[0.0001,0.01,0.1,1]
#[1, 4, 2, 11.78504375765247]

#c=[1000,2000,3000,5000,10000,0.5*1e5,1e5]
#e=[0.1,0.3,0.5,0.7,1]
#t=[0.0001,0.001,0.01,0.1,1]
#[1, 0, 2, 14.696590243254514]

#c=[1000,1500,2000,3000,10000,20000,50000,100000,200000,500000,1000000,1e7]
#e=[0,0.001,0.01,0.1,0.2,0.35,0.5,0.7,1,2,3]
#t=[0.001,0.01,0.1,1]
#[0, 9, 1, 16.867702307197174]
'''
rfe.fit(x, y)
print(rfe.ranking_)
'''

'''
#find out the best parameter
param_grid = [
{'C':[1e5,1e6,1e7,1e8,1e9],'epsilon':[0.01,0.1,0.2,0.3,0.4,0.5],
 'tol':[0.001,0.01,0.1]}]
grid_search = GridSearchCV(rfe, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(x, y)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
'''

'''
#x=x.reshape(-1,1)
#for individual data
rf.fit(x,y)
z=rf.predict(x)
print(rf.coef_)
names=np.arange(0,90,1).tolist()
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

#print(z.shape)
#train rfe

#print(rfe.grid_scores_)
#print("Features sorted by their rank(SVR):")
#print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
#print(rfe.ranking_)
#print(rf.coef_**2)


#print specific feature with target
x=x[:,[43]]
plt.plot(x, z, 'bo', label='linear model')
plt.plot(x,y,'ro')
plt.ylabel('y')
plt.xlabel('featrue')
plt.figure(1)
plt.show()
'''

#correlation
df = pd.DataFrame(x)

#Using Pearson Correlation
plt.figure(figsize=(17,17),dpi=400)
cor = df.corr()
sns.heatmap(cor, cmap='seismic')
plt.show()

#print(cor.mask)


'''
x_=range(0,180,2)
plt.plot(x_,rf.coef_**2,'ro')
plt.plot(x_,rf.coef_**2)
plt.ylabel('SVR coefficient')
plt.xlabel('triplet angle')
plt.figure(1)
plt.show()
'''