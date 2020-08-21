
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from sklearn import linear_model
#
# # Data set
# x = np.array(list(range(1, 11))).reshape(-1, 1)
# y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).ravel()
#
# # Fit regression model
# model1 = DecisionTreeRegressor(max_depth=1)
# model2 = DecisionTreeRegressor(max_depth=8)
# model3 = linear_model.LinearRegression()
# model1.fit(x, y)
# model2.fit(x, y)
# model3.fit(x, y)
#
# # Predict
# X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
# y_1 = model1.predict(X_test)
# y_2 = model2.predict(X_test)
# y_3 = model3.predict(X_test)
#
# # Plot the results
# plt.figure()
# plt.scatter(x, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue",
#          label="max_depth=1", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=3", linewidth=2)
# plt.plot(X_test, y_3, color='red', label='liner regression', linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(10 * rng.rand(160, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 2 * (0.5 - rng.rand(32))  # 每五个点增加一次噪音

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=6)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_4 = RandomForestRegressor(oob_score=True)
print(X.shape)
print(np.array(y).shape)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)
regr_4.fit(X, y)

# Predict
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)
y_4 = regr_4.predict(X_test)

print (regr_4.oob_score_)


# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
#plt.plot(X_test, y_2, color="black", label="max_depth=5", linewidth=2)
#plt.plot(X_test, y_3, color="r", label="max_depth=8", linewidth=2)
plt.plot(X_test, y_4, color="black", label="rfr", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()