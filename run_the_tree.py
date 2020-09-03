
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
data=data[:,[  1, 20, 48]]
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

print(cvtrain(splitarr,200,5,1))

'''
###looking for last feature
x_pos=[]
avgcol=[]
errcol=[]
for i in range(69):
    print('i',i)
    x = data[:,[  1, 20, 48]]
    avg,err= cvtrain(splitarr,200,5,1)
    x_pos.append(i)
    avgcol.append(avg)
    errcol.append(err)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 7), dpi=600)
axes.bar(x_pos, avgcol, yerr=errcol, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.tight_layout()
plt.savefig('i feature bar_plot_with_error_bars.png')
plt.show()
'''


