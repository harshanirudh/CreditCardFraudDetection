# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:12:34 2019

@author: Harsha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df=pd.read_csv("creditcard.csv")
df.describe()
df.groupby('Class').count()
correlation=df.corr()
cormap=sns.heatmap(correlation)
x=df.iloc[:,1:30]
y=df.iloc[:,30]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x['NormAmount']=sc.fit_transform(x['Amount'].values.reshape(-1,1))
x=x.drop(['Amount'],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#from sklearn.model_selection import StratifiedShuffleSplit
#ss=StratifiedShuffleSplit(random_state=1, n_splits=10,train_size=0.7)
#for train_i,test_i in ss.split(x,y):
#    x_train,x_test=x.iloc[train_i],x.iloc[test_i]
#    y_train,y_test=y.iloc[train_i],y.iloc[test_i]


y_train.where(y_train==0).count()
y_train.where(y_train==1).count()

import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(output_dim=15, input_dim=29 ,init='uniform' ,activation='relu'))
model.add(Dense(output_dim=15,init='uniform',activation='relu'))
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,nb_epoch=285)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

####################################################################### SMOTE ########
from imblearn.over_sampling import SMOTE
smt=SMOTE()
x_train_smote,y_train_smote=smt.fit_resample(x_train,y_train)
y_train_smote=pd.DataFrame(data=y_train_smote)
x_train_smote=pd.DataFrame(data=x_train_smote)

import keras
from keras.models import Sequential
from keras.layers import Dense
model1=Sequential()
model1.add(Dense(output_dim=15, input_dim=29 ,init='uniform' ,activation='relu'))
model1.add(Dense(output_dim=15,init='uniform',activation='relu'))
model1.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

model1.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(x_train_smote,y_train_smote,batch_size=100,nb_epoch=50)

y_pred1 = model1.predict(x_test)
y_pred1= (y_pred1 > 0.5)

pickle.dump(model1,open("C:/Users/Asus/Desktop/project/ann_swote_creditcard_better.sav","wb"))

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
from sklearn.metrics import classification_report
report=classification_report(y_test,y_pred1)
print(report)

import sklearn.metrics as metric
logloss=metric.log_loss(y_test,y_pred1)
print (logloss)

fpr,tpr,thresholds=metric.roc_curve(y_test,y_pred1)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')
plt.title("Roc curve")

precision,recall,thresholds=metric.precision_recall_curve(y_test,y_pred1)
area_under_curve=metric.auc(recall,precision)

plt.plot([0,1],[0.5,0.5],linestyle="--")
plt.plot(recall,precision,marker='.')
plt.title("Precision recall curve")
