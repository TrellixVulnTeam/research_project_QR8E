from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

print(X.shape)
print(Y.shape)
print(names)

n=np.arange(0,180,2).tolist()
# use linear regression as the model
lr = LinearRegression()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X, Y)

print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

print()
print()
rf = RandomForestRegressor()
rf.fit(X,Y)
print(rf.feature_importances_)
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))