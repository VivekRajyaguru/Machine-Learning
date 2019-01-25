#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:10:23 2019

@author: vivek
"""

# Import Created Model and Predict Result

from sklearn.externals import joblib
linear_Model = joblib.load('Linear_Regression_Model_For_Salary.sav')
linear_Model.predict(2.5)

import numpy as np
data = np.array([1, 6.5, 42.25, 274.625, 1785.062])
data = np.reshape(data, (-1, 5))
polynomial_Model = joblib.load('Polynomial_Regression_Model_For_Salary.sav')
polynomial_Model.predict(data)
