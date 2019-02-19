# Apriori Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# when no headers in dataset add header=None
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) 

# Looping through dataset transactions
# each row is added in transactions list
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    

# Train Apriori Model with dataset i.e transactions
from apyori import apriori
# min_length = min length of purchase array
# min_life = min purchase of products i.e min 3 lifts of products
# min_support = min_lift*7/total_records i.e 3*7/7500
# min_confidence = purchase confidence 
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualize Result    
results = list(rules)
