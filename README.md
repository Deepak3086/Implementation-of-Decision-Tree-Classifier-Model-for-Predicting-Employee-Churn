# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DEEPAK JG
RegisterNumber:  212224220019
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```



## Output:
Data Head:

![image](https://github.com/user-attachments/assets/20033e6a-aed0-43cd-ada9-64a4a20f20c9)


Dataset info :

![image](https://github.com/user-attachments/assets/1454af51-f4dd-455d-b428-bd107c4da084)


Null Dataset:

![image](https://github.com/user-attachments/assets/55c47028-a4ef-41ed-a77a-8c08659460c0)


Values count in left column:

![image](https://github.com/user-attachments/assets/a50548c4-0fb9-4127-8820-89d7f6ad7b07)


Dataset transformed head:

![image](https://github.com/user-attachments/assets/c61ffa24-dd5e-4473-bd73-817d70f134d3)


x.head:
![image](https://github.com/user-attachments/assets/ecdfa1e1-59ee-4746-af30-17d93a603efc)

Accuracy:

![image](https://github.com/user-attachments/assets/b4d421c4-43c7-4c20-b80f-a946fbbfc26b)


Data prediction:

![image](https://github.com/user-attachments/assets/420cb0f0-5992-4629-810f-09122fd3a085)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
