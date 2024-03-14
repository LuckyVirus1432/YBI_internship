## YBI_internship

**Title of Project: **
Wine Quality Prediction
-----------------------------------------------------------------------------------------------------------

## **Objective**
To predict the quality of the wine with the help of support vector machine.

## **Data Source**
https://raw.githubusercontent.com/YBIFoundation/Dataset/main/WhiteWineQuality.csv

## **Import Library**
import pandas as pd
import numpy as np

## **Import Data**
df = pd.read_csv(r"https://raw.githubusercontent.com/YBIFoundation/Dataset/main/WhiteWineQuality.csv", sep=';')

## **Describe Data**
df.head()

#**Information of the DataFrame**
df.info()
df.columns

## **Data Visualization**
df.describe()

## **Data Preprocessing**
df['quality'].value_counts()
df.groupby('quality').mean()

## **Define Target Variable (y) and Feature Variables (X)**
y = df['quality']
#y
y.shape
x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
x.shape
x.head()

#**Get x variable standardized**
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)
x

## **Train Test Split**
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529, test_size=0.3, stratify=y)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

## **Modeling**
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

## **Prediction**
y_pred = svc.predict(x_test)
y_pred
y_pred.shape

## **Model Evaluation**
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

## **Explaination**
This is how we created the model which checks the quality of the wine based on other parameters.
