import pickle
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, GridSearchCV, KFold
from sklearn import model_selection, metrics
from sklearn.metrics import roc_curve, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.inspection import permutation_importance
import time

data3 = np.load('/Users/weihe/Desktop/triplet_angle_with_labels.pickle', allow_pickle=True)
size = np.array(data3).shape
# average the 3sim into 1
for i in range(size[0]):
    data3[i, 0] = data3[i][0].mean(0)
# build up training set from 50
data = data3[:, 0]
target = data3[:, 1]
data = np.stack(data, axis=0)
#0 ends at 21
data = data[:, 21:90]
# data=np.delete(data,[22,23,24,25],axis=1)
#####uniquc feature by pearson[0,1,2,3,5,6,47,48,49]
#x = data[:,[0,1,2,3,5,6,47,48,49]]
x = data
y = target

# fit the data to model 100 trees not enough
# traint not used due to automated pick max_samples
rf = RandomForestRegressor(n_estimators=150, oob_score=True,
                           max_depth=6, max_features=2,
                           n_jobs=-1, bootstrap=True)

# print('here',x.shape[0])
# rf.fit(traind,traint)

# set up k-fold
kf = KFold(n_splits=5, shuffle=False)
# create array for split
splitarr = np.arange(x.shape[0])
fig1 = plt.figure()

def cvtrain(splitarr, i, j, k):
    rmse = []
    importance=[]
    correlation=[]
    collect = []
    # only use for selection
    rf = RandomForestRegressor(n_estimators=i, oob_score=True,
                               max_depth=j, max_features=k,
                               n_jobs=-1, random_state=0, bootstrap=True)
    i = 1
    for train_index, test_index in kf.split(splitarr):
        # set up train and test set
        traind = x[train_index, :]
        traint = y[train_index]
        testd = x[test_index, :]
        testt = y[test_index]
        # train and get rmse
        '''
        ####use to delete the redundant feature from pearson--for testd
        df = pd.DataFrame(testd)
        corr_col = []
        cor = df.corr()
        # function: use to find the correlated(linear) feature
        for i in range(0, len(cor.columns)):
            for j in range(i):
                # check left-down side
                # if there is one value larger than thresold
                # then record the row(feature) 0.9 is the threshold
                if abs(cor.iloc[i, j]) > 0.9:
                    colname = cor.columns[i]
                    corr_col.append(colname)
                    collect.append([i, j])

        feature_label = range(0, x.shape[1])
        train_uncor = np.delete(x, corr_col, axis=1)
        test_uncor = np.delete(x, corr_col, axis=1)
        #feature left at this fold
        feature_left = np.delete(feature_label, corr_col, axis=0)
        print(feature_left)
        traind = traind[:,feature_left]
        testd = testd[:,feature_left]
        '''

        rf.fit(traind, traint)
        y_ = rf.predict(testd)
        x_ = test_index
        '''
        # plt.subplot(2,3,i)
        # plt.plot(x_,y_,'ro')
        # plt.plot(x_,testt,'bo')
        ax1 = fig1.add_subplot(2, 3, i)
        ###for parity plot
        plt.plot(testt,y_,'ro')
        plt.plot(testt,testt,'b')
        plt.xlabel('target value')
        plt.ylabel('predict value')
        plt.title('test fold %d' %(6-i))
        #plt.show()
        '''
        predict=rf.predict(testd)
        # append rmse for each cross-validation
        rmse.append(sqrt(mean_squared_error(testt, predict)))
        df1=pd.DataFrame(testt,columns=['a'])
        df2 = pd.DataFrame(predict,columns=['b'])
        #df1=np.float64(df1)
        #df2=np.float64(df2)

        correlation.append(df1['a'].astype('float64').corr(df2['b'].astype('float64')))#get the correlation between target and caculated
        #print(df1[0].corr(df2[0]))
        #correlation.append()
        #importance.append(rf.feature_importances_)
        rmse.append(sqrt(mean_squared_error(testt, rf.predict(testd))))#get the rmse between target and caculated
        i = i + 1

    #importance=np.vstack(importance)
    #plt.show()
    #fig1.savefig('5-fold parity.png')
    return np.average(rmse),np.average(correlation)
    #return rmse,importance
    #return np.average(rmse)

'''
###new parameters
#print(cvtrain(splitarr,200,5,1))
#get 5-fold importance
ff_imp=cvtrain(splitarr,200,5,1)
ff_imp_avg=np.average(ff_imp,axis=0)
ff_imp_std=np.std(ff_imp,axis=0)
x_pos=range(21,90)

#x_pos=range(21,90)
x_pos=range(0,9)
fig, ax = plt.subplots( figsize=(15, 12), dpi=200)
ax.bar(x_pos, ff_imp_avg, yerr=ff_imp_std, align='center', alpha=1, ecolor='black', capsize=10)
#fix the position of ticks
ax.set_xticks(x_pos)
#change the name of ticks
ax.set_xticklabels([0,1,2,3,5,6,47,48,49])
plt.xlabel('features')
plt.ylabel('importance')
plt.title('importance of features([0,1,2,3,5,6,47,48,49]) rfr-5fold')
plt.savefig('importance of features([0,1,2,3,5,6,47,48,49]) rfr-5fold.png')
plt.show()

print("Features sorted by their rank(Ridge):")
print(sorted(zip(map(lambda x: round(x, 4), ff_imp_avg), x_pos,map(lambda x: round(x, 5), ff_imp_std)),reverse=True))
'''



# print(rf.get_params()) #get information about the training
# print (rf.oob_score_)

# test the model
# p=rf.predict(testd)
# print(rf.score(testd,testt))
# mset=np.square(testt-p).mean(0)
# print(mset)

#self-written grid search for parameter
#estimator
#max_depth
#max_features
#set parameters
e=[150,200,300]
d=[4,5,6,7]
f=[1,2,5]
bestpara=[1e5 for _ in range(4)]
for i in range(len(e)):
    for j in range(len(d)):
        for k in range(len(f)):
            #start = time.process_time()
            avg_rmse = 0
            for q in range(1):
                print(i,j,k)
                print('e=',e[i],'d=',d[j],'f=',f[k])
                temp_rmse, cor=cvtrain(splitarr,e[i],d[j],f[k])
                #avg_rmse=temp_rmse+avg_rmse


            avg_rmse=temp_rmse
            if bestpara[3]>avg_rmse:
                bestpara[0]=i
                bestpara[1]=j
                bestpara[2]=k
                bestpara[3]=avg_rmse
            print('current para',bestpara)
            print("current position:i:%d, j:%d, k:%d, rmse:%s, cor:%s" %(i,j,k,avg_rmse,cor))
            #end = time.process_time()
            #print('time used:', end - start)
            print()

print(bestpara)


'''
rf = RandomForestRegressor( oob_score=True,
                           n_jobs=-1, bootstrap=True)

#find out the best parameter
param_grid = [
    {'n_estimators':[100,150,200,500],
    'max_depth':[5,6,7,8,None], 'max_features':[1,2,5,10,30,60,90,'log2','sqrt']}]
grid_search = GridSearchCV(rf, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(x, y)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
'''

'''
#use to plot importance
rf.fit(x, y)
##normal importance
importance = rf.feature_importances_
##permutation importance
pimportance = permutation_importance(rf, x, y)
pimportance = pimportance.importances_mean

themax = np.max(importance)
p = np.where(importance == themax)
print('max: ', themax)
print('loaction of max: ', p)
'''

'''
def plot_imp(importance):
    #plot the importance
    x_=range(0,180,2)
    print("(Importance,Features) sorted by their rank(rfr):")
    resortedimp=sorted(zip(map(lambda x_: round(x_, 4), importance),range(0,90,1)),reverse=True)
    sortedimp=sorted(zip(map(lambda x_: round(x_, 4), importance),range(0,90,1)),reverse=False)
    print(resortedimp)
    collect_importance=[]
    collect_label=[]
    for i in range(x.shape[1]):
        collect_importance.append(resortedimp[i][0])
        collect_label.append(resortedimp[i][1])

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (18,7), dpi=600)
    plt.bar(range(len(collect_importance)), collect_importance,tick_label = collect_label)
    plt.ylabel('importance')
    plt.xlabel('triplet angle feature')
    plt.title('random forest regreesion')
    plt.show()
    fig.savefig('rfr_importance(100,8,2).png')

#plot_imp(importance)
'''

'''
x = range(0, 180, 2)
# plot specific range of data
plt.plot(x, importance, 'ro')
plt.plot(x, importance)
plt.ylabel('importance in random forest-regression')
plt.xlabel('triplet angle')
plt.figure(1)
plt.show()
x_ = range(0, 90, 1)
print("Features sorted by their rank(Ridge):")
print(zip(map(lambda x: round(x, 4), importance), x_))
'''

'''
##permutation importance
themax=np.max(pimportance)
p=np.where(pimportance==themax)
print('max: ',themax)
print('loaction of max: ',p)
#plot the importance
x=range(0,180,2)
plt.plot(x,pimportance,'ro')
plt.plot(x,pimportance)
plt.ylabel('importance in random forest-regression')
plt.xlabel('triplet angle')
plt.figure(2)
plt.show()
'''

"""
#plot the tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
for i in range(0,10):
    tree.plot_tree(rf.estimators_[i],
                   filled=True);
    fig.savefig(f'rfr_individualtree{i}.png')

"""

"""
tree.plot_tree(rf.estimators_[100],
               filled = True);
fig.savefig('rfr_individualtree0.png')
###tree only gives distinct data points, so can be different
tree.plot_tree(rf.estimators_[1],
               filled = True);
fig.savefig('rfr_individualtree1.png')
"""

