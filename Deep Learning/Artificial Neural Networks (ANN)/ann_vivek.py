# ANN

# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Dummy Variables for Country Category
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Remove 1 Dummy Varible Column for Dummy Variable Trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Prepare ANN
import keras
from keras.models import Sequential # initialize neural network
from keras.layers import Dense # for Layers

# Initialize ANN

classifier = Sequential()

# Adding Input Layer and Hidden Layer
classifier.add(Dense(units=6,activation='relu', input_dim= 11))
# Add Second Hidden Layer
classifier.add(Dense(units=6,activation='relu'))
# Add output Layer
classifier.add(Dense(units=1,activation='sigmoid'))

# Compile ANN
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit ANN to Training Set
classifier.fit(X_train, y_train, epochs=100, batch_size=10) # Acc - 0.8649, Loss - 0.3281

# Making Predections 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# 85% Accuracy 