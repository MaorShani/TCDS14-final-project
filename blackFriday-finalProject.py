# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:26:59 2020

@author: maors
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as pyplot
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Read DataSet

data=pd.read_csv('C:/Users/maors/Documents/Maor/Data Science/Final Project/Data Sets/train.csv')

##### Read Data From Daniella Computer
#data=pd.read_csv('C:/Users/USER/Desktop/bfsp1.csv')

# Data Exploration

data.info()
data.head()
data.describe()
data['User_ID'].nunique()
data['Product_ID'].nunique()

# data Exploration
sns.countplot(x ='Gender', data = data)
plt.show()

# Remove unnecesary columns

data=data.drop(['User_ID','Product_ID'], axis=1)

# Split the 3 multile choice categories columns to 20 distinct columns

cat1=pd.get_dummies(data['Product_Category_1'],dummy_na=False)
cat2=pd.get_dummies(data['Product_Category_2'],dummy_na=False)
cat3=pd.get_dummies(data['Product_Category_3'],dummy_na=False)

pro_cat=pd.concat([cat1,cat2,cat3]).groupby(level=0).any().astype(int)

### Some temporary data exploration
#pro_cat.head(30)
#pro_cat_head= pro_cat.head(10)
data_head =data.head(30)
#data_describe = data.describe()

# Split the data set to X and Y. X currently include only the user features

data_x = data[['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']]
data_x = data_x.astype('category')
data_y = data[['Purchase']]

# Set ALL of the user features as dummy variables.
data_x = pd.get_dummies(data=data_x, drop_first=True)
data_x = pd.concat([data_x,pro_cat], axis=1)

# Split the data to train and test

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)

### Temp variable
data_train_x_head = data_train_x.head(10)

regr = linear_model.LinearRegression()
regr.fit(data_train_x, data_train_y)
#print(regr.coef_)
print(f'R2 is {regr.score(data_train_x, data_train_y)}')
