# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:57:00 2020

@author: maors
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv('C:/Users/maors/Documents/Maor/Data Science/Final Project/Data Sets/data.csv')

data=data.drop(['User_ID','Product_ID'], axis=1)

data = pd.melt(data,id_vars=['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Purchase'],var_name="temp",value_name='Category')
data = data.drop(['temp','Purchase'], axis = 1)
data = data[data['Category'].notna()]

data_head = data.sample(1000)

data_x = data[['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']]
data_x = data_x.astype('category')
data_y = data[['Category']]

data_y_head = data_y.sample(1000)


data_x = pd.get_dummies(data=data_x, drop_first=True)

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)

data_y_head = data_test_y.sample(1000)
data_train_x_head = data_train_x.head(30)

tree = DecisionTreeClassifier(criterion="gini")
tree.fit(data_train_x, data_train_y)
prediction = tree.predict(data_test_x)

confusion_table = confusion_matrix(data_test_y, prediction)
print(f'accuracy score is {accuracy_score(data_test_y, prediction)}')
print(" Decision Tree Report:\n",classification_report(data_test_y, prediction))


nb = GaussianNB()
nb.fit(data_train_x, data_train_y)
prediction = nb.predict(data_test_x)

confusion_table2 = confusion_matrix(data_test_y, prediction)
accuracy_score(data_test_y, prediction)
print(f'accuracy score is {accuracy_score(data_test_y, prediction)}')

print(" Naive Base Report:\n",classification_report(data_test_y, prediction))

# data_combined = pd.concat([data_test_y,pd.DataFrame(prediction)],axis=1)
# data_combined = data_test_y
# data_combined['prediction']=pd.Series(prediction,index = data_combined.index)
# data_combined_head = data_combined.head(100)
# data_test_y.groupby('Category').count()

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(data_train_x, data_train_y)
prediction = knn.predict(data_test_x)
onfusion_table3 = confusion_matrix(data_test_y, prediction)
accuracy_score(data_test_y, prediction)
print(f'accuracy score is {accuracy_score(data_test_y, prediction)}')
print(" KNN Report:\n",classification_report(data_test_y, prediction))