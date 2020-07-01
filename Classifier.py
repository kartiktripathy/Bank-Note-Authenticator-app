#Classification Template

#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('BankNote_Authentication.csv')

#Separating the dependant and independant variable
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#Splitting the dataset into the training and test dataset
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#Fitting Classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf' , random_state=0)
classifier.fit(X_train,Y_train)

#Predicting The Test Results
y_pred = classifier.predict(x_test)

#Making the confusion MAtrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)
print("Confusion Matrix : ")
print(cm)
print("Accuracy : ",ac*100,"%")

import pickle
pickle.dump(classifier , open('note_authenticator.pkl','wb'))