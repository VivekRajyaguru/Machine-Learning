# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting the Logistic Regression Model to the dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)


# Predicting a new result
y_pred = classifier.predict(X_test)


# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the Training set Result
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualize the Testing set Result
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

print(X_set[:, 0].min() - 1) # -2.9931891594584856
print(X_set[:, 0].max() + 1) # 3.1568108405414717
print(X_set[:, 1].min() - 1) # -2.5825424477554764
print(X_set[:, 1].max() + 1) # 3.3315320031817324


grid_1 = np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1) #615
grid_2 = np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01) #591

'''count = -2.9931891594584856
for i in range(1, 616, 1):
    count = count + 0.01
print(count) '''

mesgrid = np.meshgrid(grid_1,grid_2)
ravel_1 = mesgrid[0].ravel()
ravel_2 = mesgrid[1].ravel()
print(ravel_1.shape)
print(ravel_2.shape)
np_array = np.array([ravel_1, ravel_2]).T
predict_array = classifier.predict(np_array)
predict_array.reshape(ravel_1.shape)
print(predict_array.shape)
print(np.unique(y_set))
print(enumerate(np.unique(y_set)))
X_set_0 = X_set
X_set_1 = X_set
for i, j in enumerate(np.unique(y_set)):
    print(i, j)
    if i == 0:
       print(X_set[y_set == 0, 1])
       print(X_set[y_set == 0, 0])
       X_set_0 = X_set[y_set == j, 0]
    else:
        X_set_1 = X_set[y_set == j, 1]
        
colorMap = ListedColormap(('red', 'green'))
print(colorMap(0))
print(colorMap(1))

X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

