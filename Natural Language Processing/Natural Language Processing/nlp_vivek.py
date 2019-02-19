# NLP Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Split by Tab in tsv File, to ignore double quote add 3
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting = 3)

# Cleaning Texts
import re
import nltk
# download stop words to remove from review
nltk.download('stopwords')
from nltk.corpus import stopwords

# importing stamming library
from nltk.stem import PorterStemmer

# remove extra character and only kept Alphabets in review for first review
review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][0])

# converting all characters in lowercase
review = str(review).lower()

# remove unwanted words from review
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))] 

# stamming process to only keep root of word like update loved to love    
stemmer = PorterStemmer()
review = [stemmer.stem(word) for word in review] 

# joing review from array to string
review = ' '.join(review)


# Cleaning for all Reviews
corpus = []
for i in range(0, 1000):
    # remove extra character and only kept Alphabets in review for first review
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    
    # converting all characters in lowercase
    review = str(review).lower()
    
    # remove unwanted words from review
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))] 
    
    # stamming process to only keep root of word like update loved to love    
    stemmer = PorterStemmer()
    review = [stemmer.stem(word) for word in review] 
    
    # joing review from array to string
    review = ' '.join(review)
    corpus.append(review)
    

# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray() # Sparse Matrix
y = dataset.iloc[:, 1].values  # Liked Result


# Evaluating With Different Classification Models

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# With Naive bayes = 73% Accuracy

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
   

# With Random Forest = 72% Accuracy

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# With Kernel SVM = 49% Accuracy
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# With KNN = 62% Accuracy
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,p=2, metric='minkowski')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
   


