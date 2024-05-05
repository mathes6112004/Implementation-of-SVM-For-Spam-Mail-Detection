# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step-1: start.
step-2: Import chardet.
step-3: Read the dataset.
step-4: Import SVC from sklearn.
step-5: Fit the data in the model and run the algorithm.
step-6: stop.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Matheswaran
RegisterNumber:  212222110024

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
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
##### data.head():
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/79331dd9-d5c6-4c1e-9deb-00120890679d)

##### data.tail():
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/8929ac18-aba8-4c68-be92-b70ac5631c8c)

##### data.info():
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/5b9fc559-8723-419a-ad20-91590c63d4da)

##### data.isnull().sum():
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/a41f7ab0-c057-40a3-8933-b77440b66b86)

##### Y_prediction value:
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/0436e853-ca92-49f3-b48a-3a4add28ba7f)

##### Accuracy value:
![image](https://github.com/mathes6112004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477782/c6d4f9e5-7d6a-4bb6-b233-ff625a8e615b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
