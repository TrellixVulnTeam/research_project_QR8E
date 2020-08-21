from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
data=[[0,0,0],[1,1,1],[2,2,2],[1,1,1],[2,2,2],[0,0,0]]
target=[0,1,2,1,2,0]
rf = RandomForestRegressor()
rf.fit(data, target)
print(np.shape(data))
#print(rf.predict([[1,1,1]]))
#print(rf.predict([[1,1,1],[2,2,2]]))
#[ 1.]
#[ 1.  1.9]

#data2=[[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
data2=[[3],[6],[9],[10],[15],[19]]

target2=[0,1,2,3,4,5]
rf2 = RandomForestRegressor()
rf2.fit(data2, target2)
rf3 = RandomForestRegressor()
rf3.fit(data2, target2)
rf4 = RandomForestRegressor()
rf4.fit(data2, target2)
rf5 = RandomForestRegressor()
rf5.fit(data2, target2)

print(rf2.predict([[3]]))
#print(rf2.predict([[1,1,1],[2,2,2],[4,4,4]]))
print(rf2.feature_importances_)
result=rf2.predict(data2)
print(result)


plt.figure()
plt.scatter(data2, target2, edgecolor="black",
            c="darkorange", label="data")
plt.plot(data2,result, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.plot(X_test, y_3, color="r", label="max_depth=8", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


#from sklearn.tree import export_graphviz
#export_graphviz(tree_in_forest,
#                feature_names=data2.columns,
#                filled=True,
#                rounded=True)
#os.system('dot -Tpng tree.dot -o tree.png')



#print(1/4*(rf2.predict([[1,1,1]])+rf3.predict([[1,1,1]])+rf4.predict([[1,1,1]])+rf5.predict([[1,1,1]])))
#[ 0.7]
#[ 0.7  1.8  4. ]
