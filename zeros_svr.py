import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sns as sns
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
#get rid of 0
data = data[:, 21:90]
x_temp=data;
y=target;
#let the print print all the data, no ....
#np.set_printoptions(threshold=np.inf)
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

'''
#normalized
#min-max normalization for feature
tmax=np.max(x_temp)
tmin=np.min(x_temp)
x=(x_temp-tmin)/(tmax-tmin)
x=x[:,:]
'''

#normalized (x-xo)/v
x_avg=np.average(x_temp,axis=1)
x_std=np.std(x_temp,axis=1)
x_temp=x_temp-x_avg[:,None]
x=x_temp/x_std[:,None]



##x=x_temp
#for i in range(0,58):
#    plt.plot(names,x[i,:])
#plt.show()

#for test
#train rfsvr
#rf=SVR(kernel='linear',C=1e3,epsilon=10,max_iter=-1)
#use linear svr---need cv matrix
#need to choose variable
#rf=LinearSVR(C=10000, epsilon=0.2, max_iter=10000000.0, tol=0.1)
#rf=LinearSVR(max_iter=10000000.0)
#print(x.shape)

#rfe = RFE(rf,n_features_to_select=1)
#set up k-fold
kf = KFold(n_splits=5, shuffle=False)
#create array for split
splitarr = np.arange(x.shape[0])
#the_ranking=[]
print('here')

#add the version of ploting
def cvtrain(splitarr,i,j,k):
    rmse = []
    #for each train set get one linearsvr
    rf = LinearSVR(C=i, epsilon=j, tol=k, max_iter=10000000.0)
    rfe = RFE(rf, n_features_to_select=1)
    for train_index , test_index in kf.split(splitarr):

        #set up train and test set
        traind=x[train_index,:]
        traint=y[train_index]
        testd=x[test_index,:]
        testt=y[test_index]
        #print(traind.shape)
        #train and get rmse
        rfe.fit(traind, traint)
        rmse.append(sqrt(mean_squared_error(testt, rfe.predict(testd))))
        names = np.arange(0, 90, 1).tolist()
        #the_ranking.append(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

    #return [rmse,the_ranking]
    print(rmse)
    return np.average(rmse)
'''
rf=LinearSVR(C=50000,epsilon=0.2,tol=0.1,max_iter=1000000)
print(cvtrain(splitarr,5000,0.2,0.1))
rfe = RFE(rf, n_features_to_select=1)
rfe.fit(x,y)
names=np.arange(0,90,1).tolist()
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
'''


'''
#verion used to subplot
#draw the graph with prediction 
def cvtrain(splitarr,i,j,k):
    rmse = []
    fig1 = plt.figure()
    #for each train set get one linearsvr
    rf = LinearSVR(C=i, epsilon=j, tol=k, max_iter=10000000.0)
    rfe = RFE(rf, n_features_to_select=1)
    t=1
    for train_index , test_index in kf.split(splitarr):
        #set up train and test set
        traind=x[train_index,:]
        traint=y[train_index]
        testd=x[test_index,:]
        testt=y[test_index]
        #train and get rmse
        rfe.fit(traind, traint)
        y_ = rfe.predict(testd)
        x_ = test_index
        fig1.add_subplot(2, 3, t)
        plt.plot(x_,y_,'ro')
        plt.plot(x_,testt,'bo')
        names = np.arange(0, 90, 1).tolist()
        rmse.append(sqrt(mean_squared_error(testt, rfe.predict(testd))))
        #the_ranking.append(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
        t = t + 1
    #return [rmse,the_ranking]
    plt.show()
    fig1.savefig('5-fold prediction-svr.png')
    #print(rmse)
    return np.average(rmse)
'''


#print(cvtrain(splitarr,5000,0.2,0.1))

#print(cvtrain(splitarr,5000,0.2,0.1))
#print(cvtrain(splitarr,10000,0.2,0.1))
#print(cvtrain(splitarr,20000,0.2,0.1))
#print(cvtrain(splitarr,50000,0.2,0.1))

###########print(cvtrain(splitarr,5000,0.5,0.1))#
#print(cvtrain(splitarr,10000,0.5,0.1))
#print(cvtrain(splitarr,20000,0.5,0.1))
#print(cvtrain(splitarr,50000,0.5,0.1))
#print(cvtrain(splitarr,100000,0.5,0.1))

#print(cvtrain(splitarr,5000,1,0.1))#
#print(cvtrain(splitarr,10000,1,0.1))
#print(cvtrain(splitarr,20000,1,0.1))
#print(cvtrain(splitarr,50000,1,0.1))

#print(cvtrain(splitarr,5000,0.5,0.01))
#print(cvtrain(splitarr,10000,0.5,0.01))
#print(cvtrain(splitarr,20000,0.5,0.01))
#print(cvtrain(splitarr,50000,0.5,0.01))



#print(cvtrain(splitarr,1e4,0.2,0.1))
#print("rmse:",rmse)
#print("sum rmse:",np.sum(rmse))

#[21.641898635629975, 8.062483069218233, 14.13362811214451, 21.780609726322155, 14.68078345484246]
#[22.064243442051257, 10.274495339594454, 16.97822463589625, 24.46693640421105, 15.409378255174952]

'''
df = pd.DataFrame(x)
#Using Pearson Correlation
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,15), dpi=300)
cor = df.corr()
sns.heatmap(cor, cmap='seismic',linecolor='black',linewidths='0.1')
plt.show()
fig.savefig('correlation.png')
'''

'''
#training-test part
rf = LinearSVR(C=5000, epsilon=0.2, tol=0.1, max_iter=10000000.0)
rfe = RFE(rf, n_features_to_select=1)

rfe.fit(x, y)
print(rfe.ranking_)
names=np.arange(0,69,1).tolist()
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
'''





#222
#self-written grid search for parameter
c=[1000,2000,5000,10000,5*1e4,1e5,1e6,1e7]
e=[0,0.1,0.2,0.5,1,2]
t=[0.001,0.01,0.1]
bestpara=[1e5 for _ in range(4)]
for i in range(len(c)):
    for j in range(len(e)):
        for k in range(len(t)):
            start = time.process_time()
            avg_rmse = 0
            for q in range(1):
                temp_rmse=cvtrain(splitarr,c[i],e[j],t[k])
                #avg_rmse=temp_rmse+avg_rmse
            avg_rmse=temp_rmse
            if bestpara[3]>avg_rmse:
                bestpara[0]=i
                bestpara[1]=j
                bestpara[2]=k
                bestpara[3]=avg_rmse
            print('current best para position',bestpara)
            print("current position:i:%d, j:%d, k:%d, rmse:%s" %(i,j,k,avg_rmse))
            print('current parameter',c[i],e[j],t[k])
            end = time.process_time()
            print('time used:', end - start)
            print()

print(bestpara)


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

#c=[1000,5000,1e4,5*1e4,1e5,1e6,1e7]
#e=[0.1,0.2,0.5,0.7,1,2]
#t=[0.0001,0.001,0.1]
#[2, 1, 2, 16.1327625343376]


'''
#####use to get the result
rf = LinearSVR(C=5000, epsilon=0.2, tol=0.1, max_iter=10000000.0)
rfe = RFE(rf, n_features_to_select=1)

rfe.fit(x, y)
print(rfe.ranking_)
names=np.arange(0,90,1).tolist()
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
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
'''

#print(z.shape)
#train rfe

#print(rfe.grid_scores_)
#print("Features sorted by their rank(SVR):")
#print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
#print(rfe.ranking_)
#print(rf.coef_**2)

'''
#print specific feature with target
x=x[:,[55]]
plt.plot(x, z, 'bo', label='linear model')
plt.plot(x,y,'ro')
plt.ylabel('y')
plt.xlabel('featrue')
plt.figure(1)
plt.show()
'''


'''
x_=range(0,180,2)
plt.plot(x_,rf.coef_**2,'ro')
plt.plot(x_,rf.coef_**2)
plt.ylabel('SVR coefficient')
plt.xlabel('triplet angle')
plt.figure(1)
plt.show()
'''
#c=[1000,2000,3000,5000,10000,0.5*1e5,1e5]
#e=[0.1,0.3,0.5,0.7,1]
#t=[0.0001,0.001,0.01,0.1,1]
#[1, 0, 3, 14.6