# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:23:51 2018

@author: Anil
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data
dataset = pd.read_csv('Data.csv')
indep = dataset.iloc[:, :-1].values
dep = dataset.iloc[:, 3].values

#Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(indep[:, 1:3])
indep[:, 1:3] = imputer.transform(indep[:, 1:3])

#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_indep = LabelEncoder()
indep[:, 0] = labelencoder_indep.fit_transform(indep[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
indep = onehotencoder.fit_transform(indep).toarray()
labelencoder_dep = LabelEncoder()
dep = labelencoder_dep.fit_transform(dep)

#Splitting Data for Training and Testing
from sklearn.model_selection import train_test_split
indep_train, indep_test, dep_train, dep_test = train_test_split(indep, dep, test_size = 0.2, random_state = 0)

#Scaling
from sklearn.preprocessing import StandardScaler
sc_indep = StandardScaler()
indep_train = sc_indep.fit_transform(indep_train)
indep_test = sc_indep.fit_transform(indep_test)