# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. 4.calculate Mean square error,data prediction and r2.



## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: vinush.cv  
RegisterNumber:  212222230176
*/


import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head():

![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/8658a7c9-3bfe-436c-82af-efcedacb4a85)


## data.info:


![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/a80c5933-3ea0-46b5-b26b-7391a47397b0)


## isnull() and sum():


![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/c8b075eb-2daf-48f4-b8a7-d447cba0d5a5)


## MSE value:


![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/704e3b48-366e-4a53-8afc-3fa28322b9b9)


## r2 value;

![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/7404e3be-e0c2-4c44-b2b9-45702d025928)


## data prediction:


![image](https://github.com/vinushcv/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113975318/affcfc7b-1742-4d31-a8f1-49c93fc3c801)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
