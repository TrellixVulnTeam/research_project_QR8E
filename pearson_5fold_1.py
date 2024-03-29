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
data3=np.load('/Users/weihe/Desktop/triplet_angle_with_labels.pickle',allow_pickle=True)
size=np.array(data3).shape
#average the 3sim into 1
for i in range(size[0]):
    data3[i,0]=data3[i][0].mean(0)

#print(np.array(data3[0,0]).shape)
#build up training set from 50
data=data3[:,0]
target=data3[:,1]
data=np.stack( data, axis=0 )

x_temp=data;
y=target;

#normalized
#min-max normalization for feature
#no normalized in correlation
#tmax=np.max(x_temp)
#tmin=np.min(x_temp)
#x=(x_temp-tmin)/(tmax-tmin)
#x=x[:,:]
x=x_temp
x=x[:,21:90]
kf = KFold(n_splits=5, shuffle=False)
#create array for split
splitarr = np.arange(x.shape[0])
collect=[]
search1=[43, 76,25,77,83,24,87,82,80,46,45,74,42,57,64,73,69,65,58,70,22,75,28,44,33,79,78,62,89,86]#start
search2=[24, 47,23, 69, 22, 26,27,68]#target
df=pd.DataFrame(x)
#print(df.iloc[2:23])

# Using Pearson Correlation
###fig, axes = plt.subplots(nrows=i, ncols=1, figsize=(15, 15), dpi=300)
corr_col=[]
####--all featrue list
l=[]
for i in range(69):
    l.append(i)
l2=[str(x) for x in l]
df.columns=l2
#set the correlated data
#cor = df[['43'], '76','25','77','83','24','87','82','80','46',
#         '45','74','42','57','64','73','69','65','58','70','22','75'].corr(df['24','47','23', '69', '22', '26','27','68'])
cor=df[[ '1', '2', '15', '20','28','47','48','49']].corr()
print(cor)
#get the colomn
####cor_now=cor.loc[['0', '1', '2', '15', '20','28','47','48','49']]
####print(cor_now)
#get the row
####cor_now=cor_now.loc[['0', '1', '2', '15', '20','28','47','48','49']]
####p=cor_now.values.reshape(10,10)
####sns.heatmap(p,  linecolor='black', linewidths='0.01',)
sns.heatmap(cor,  linecolor='black', linewidths='0.01',annot = True)
plt.show()



for train_index , test_index in kf.split(splitarr):
    i=1
    #set up train and test set
    #training data and target
    traind=x[train_index,:]
    traint=y[train_index]
    #testing data and target
    testd=x[test_index,:]
    testt=y[test_index]
    #train and get rmse
    df = pd.DataFrame(testd)
    # Using Pearson Correlation
    ###fig, axes = plt.subplots(nrows=i, ncols=1, figsize=(15, 15), dpi=300)
    corr_col=[]
    #set the correlated data
    cor = df.corr()

    '''
    #function: use to find the correlated(linear) feature
    for i in range(0,len(cor.columns)):
        for j in range(i):
            #check left-down side
            #if there is one value larger than thresold
            #then record the row(feature) 0.9 is the threshold
            if abs(cor.iloc[i,j])>0.9:
                colname = cor.columns[i]
                corr_col.append(colname)
                collect.append([i,j])

    feature_label=range(0,x.shape[1])
    train_uncor=np.delete(traind,corr_col,axis=1)
    test_uncor=np.delete(traind,corr_col,axis=1)
    feature_left=np.delete(feature_label,corr_col,axis=0)
    #print(corr_col)
    print(feature_left)
    '''
    ##
    #sns.heatmap(cor, cmap='seismic', linecolor='black', linewidths='0.1')
    #plt.show()
    #fig.savefig('correlation.png')
    i=i+1



'''
#####used for breath first search 
#collect=np.vstack(collect)
df = pd.DataFrame(x)
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

feature_label=range(0,x.shape[1])
train_uncor=np.delete(x,corr_col,axis=1)
test_uncor=np.delete(x,corr_col,axis=1)
feature_left=np.delete(feature_label,corr_col,axis=0)
search1=[43, 76,25,77,83,24,87,82,80,46,45,74,42,57,64,73,69,65,58,70,22,75,28,44,33,79,78,62,89,86]#start
search2=[24, 47,23, 69, 22, 26,27,68]#target
search1[:] = [q - 21 for q in search1]
search2[:] = [q - 21 for q in search2]


# find the link between start and target
# s is iterate at each time
# t is target
# collect is the map (int,list,list)
# chain bfs
correlation=[]
#find e in list
def inlist(e,list):
    for i in range(len(list)):
        if e == list[i]:
            return True
    return False

def findcor(s,t,collect):
    temp = []
    mask = [0] * 69
    #mark for the begining -1
    temp.append([-1,s])
    mask[s] = 1
    while len(temp)!=0:
        #print(temp)
        curr=temp.pop(0)
        if inlist(curr[1], t):
            #if found, return
            return curr[1]
        for i in range(len(collect)):
            #find correlation and check if used
            if collect[i][0] == curr[1] and mask[collect[i][1]]==0:
                temp.append(collect[i])
                #set used once in the temp
                mask[collect[i][1]] = 1;


#8 distinct featrues
result=[]
for a in range(len(search1)):
    for b in range(len(search2)):
        #print(collect)
        #print(findcor(search1[a],search2,collect)+21)
        temp=[]
        temp.append(search2[b])
        temp1=findcor(search1[a], temp, collect)
        if temp1!=None:
            temp1=temp1+21
        #temp1=findcor(search1[a], temp, collect) + 21
        result.append(temp1)

y=np.array(result)
y=y.reshape(-1,8)
print(y)
'''

#print(corr_col)
#print(feature_left)
#sns.heatmap(cor, cmap='seismic', linecolor='black', linewidths='0.1')
#plt.show()
