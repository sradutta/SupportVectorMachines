#importing the iris datasets
from sklearn import datasets
iris = datasets.load_iris()

#plot petal length and sepal width of the three types of flowers
import matplotlib.pyplot as plt
plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

#choosing only the first 100 flowers that contains only two species. c=iris.target tells which flower to choose. 0 = setosa, 1 = versicolor, 2 = virginica
#plotting sepal width, petal length
plt.scatter(iris.data[0:100,1], iris.data[0:100,2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

#plotting all the 4 characteristics of setosa which is contained in lines 0--50
plt.scatter(iris.data[0:50,0],iris.data[0:50,1], c=iris.target[0:50])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[0:50,2], iris.data[0:50,3], c=iris.target[0:50])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

#plotting all the 4 characteristics of versicolor which is contained in lines 51--100
plt.scatter(iris.data[51:100,0],iris.data[51:100,1], c=iris.target[51:100])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[51:100,2], iris.data[51:100,3], c=iris.target[51:100])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


#plotting all the 4 characteristics of virginica which is contained in lines 101--150
plt.scatter(iris.data[101:150,0],iris.data[101:150,1], c=iris.target[101:150])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[101:150,2], iris.data[101:150,3], c=iris.target[101:150])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

#plotting sepal length, sepal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

#plotting petal length, petal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

'''yet to find any combination that separates the flowers clearly. But, I've not done all the combinations as there are too many to do.'''



#apply the SVC module to the first two flowers -- serasota and versicolor contained in 0--100
from sklearn import svm
svc = svm.SVC(kernel='linear')
X = iris.data[0:100, 1:3] #choose col 1 and 2 for rows 0--100
y = iris.target[0:100] #choose only the first two flowers which are in 0--100
svc.fit(X,y)

#visualizing the above results
#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
import numpy as np

#ListedColormap is used for generating a custom colormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#meshgrid -- given a 1d vector, it returns a matrix of size specified
#linspace -- returns evenly spaced numbers over a specified interval. Thus, in the example below, it's returning a 1d array of numbers of size 100 that lies between x_min and x_max
#ravel -- returns a flattened array. Thus a 2d array [[1,2, 3], [4,5,6]] is returned as [1,2, 3, 4, 5, 6]
#reshape -- gives a new shape to an array without changing its data
def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
plot_estimator(svc, X, y)
plt.show()


#apply the SVC module to all the flowers
X = iris.data[0:, 1:3] #choose col 1 and 2 for all the rows 
y = iris.target[0:] #choose all the flowers 
svc.fit(X,y)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
plot_estimator(svc, X, y)
plt.show()

'''the code is not much different when three flowers are classified. The margins and the line changes based on x_min, x_max, y_min, y_max and if we are using 100 or 500 or other numbers for meshgrid.'''
