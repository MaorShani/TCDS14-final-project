# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:26:59 2020

@author: maors
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import matplotlib as pyplot
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

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
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Count rows - grouped by features\' values')
sns.countplot(ax=axes[0,0],x ='Gender', data = data)
sns.countplot(ax=axes[0,1],x ='Age', data = data, order = ["0-17", "18-25","26-35","36-45","46-50","51-55","55+"])
sns.countplot(ax=axes[0,2],x ='Marital_Status', data = data)
sns.countplot(ax=axes[1,0],x ='Occupation', data = data)
sns.countplot(ax=axes[1,1],x ='City_Category', data = data, order=["A","B","C"])
sns.countplot(ax=axes[1,2],x ='Stay_In_Current_City_Years', data = data, order = ["0","1","2","3","4+"])

plt.show()

plt.hist(data['Purchase'])
plt.xlabel("Purchase")
plt.ylabel("Count")
plt.title("Purchace Histogram")

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

# Linear Regression

regr = linear_model.LinearRegression()
regr.fit(data_train_x, data_train_y)
#print(regr.coef_)
print(f'R2 is {regr.score(data_train_x, data_train_y)}')
prediction = regr.predict(data_test_x)
mse =  mean_squared_error(data_test_y, prediction)
print(f'MSE is {mse}')

# Random Forest

data_train_y_np = np.array(data_train_y)
data_train_x_np = np.array(data_train_x)

rf = RandomForestRegressor(max_depth = 5)
rf.fit(data_train_x_np, data_train_y_np)
prediction_rf = rf.predict(data_test_x)
mse_rf =  mean_squared_error(data_test_y, prediction_rf)
print(f'Random_Forest MSE is {mse_rf}')

fn=data_train_x.columns.values.tolist()
cn=data_train_y.columns.values.tolist()
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[5],
               feature_names = fn, 
               class_names=cn,
               filled = True);
#fig.savefig('C:/Users/maors/Documents/Maor/Data Science/Final Project/rf_individualtree.png')

importances = list(rf.feature_importances_)
feature_importances = [(data_train_x, round(importance, 4)) for data_train_x, importance in zip(fn, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Linear Regression using statmodel in order to get p-values

data_train_x2 = sm.add_constant(data_train_x)
model = sm.OLS(data_train_y,data_train_x2)
model2 = model.fit()
print(model2.summary())

# Removing unimportant columnns based on coefficient p-vlues

data_train_x_reduced = data_train_x.drop(columns=['Gender_M',
                                                  'Stay_In_Current_City_Years_1',
                                                  'Stay_In_Current_City_Years_2',
                                                  'Stay_In_Current_City_Years_3',
                                                  'Stay_In_Current_City_Years_4+'])

data_test_x_reduced = data_test_x.drop(columns=['Gender_M',
                                                  'Stay_In_Current_City_Years_1',
                                                  'Stay_In_Current_City_Years_2',
                                                  'Stay_In_Current_City_Years_3',
                                                  'Stay_In_Current_City_Years_4+'])

regr = linear_model.LinearRegression()
regr.fit(data_train_x_reduced, data_train_y)
#print(regr.coef_)
print(f'New R2 is {regr.score(data_train_x_reduced , data_train_y)}')
prediction = regr.predict(data_test_x_reduced)
mse =  mean_squared_error(data_test_y, prediction)
print(f'new MSE is {mse}')

