# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook.

## Algorithm:
```
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
```
## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by:Murali Krishna S 
RegisterNumber:212223230129
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/040178f5-ebb1-4651-bfe7-b7c289c7dbf4)
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/aec1da9c-f920-48b1-8888-da48224d0cc5)
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/0537c46c-d9fc-4b70-9068-97bb1b5f96ce)
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/8289430f-673e-4c21-bbf2-de3efc5c578e)
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/77ac77f8-6008-4b3d-8b70-dae665ed2e4b)
![image](https://github.com/Murali-Krishna0/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149054535/2ce3655e-aab2-4233-9ba4-2ab811d4b1d0)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
