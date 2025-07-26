#Step 1:  Importing required libraries:
import pandas as pd  # used for data reading and handling {as popular}
import numpy as np  #  for number operation
import matplotlib.pyplot as plt  # ploting graphs 
import seaborn as sns  # Advanced visualization  
from sklearn.model_selection import train_test_split  #  spliting data for testing and training
from sklearn.preprocessing import LabelEncoder  #  used to convert categorical data into numeric value
from sklearn.ensemble import RandomForestClassifier  # developmet of model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  #  Evaluating model

#Step 2:  Dataset Reading
data = pd.read_csv("loan_prediction.csv")  #  Reading CSV file

#Step 3:  Dataset Operation
print("First 5 rows of data:\n", data.head())  # highlighting first 5 rows[
print("\nData types and missing values:\n")
print(data.info())  #  Checking col type and value
print("\nhow much missing values are there:\n", data.isnull().sum())

#Step 4: Filling missing value by mathematical operations]
# Mode:frequently
 
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])#filling empty value here by mode estimation 
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())#filling empty value here by median estimation


# Median: Mid-one
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())#filling it by median for an estimation


#Step 5:  Removing irrelevant col.
data.drop('Loan_ID', axis=1, inplace=True)

#Step 6: Converting Categorical val into numeric 
# Assign number to every string
le = LabelEncoder()
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols:
    data[col] = le.fit_transform(data[col])

#Step 7:  replace 3+ by 3 and then converted into int
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].astype(int)

#Step 8:  seperating features and targets
X = data.drop('Loan_Status', axis=1)  # X=input features
y = data['Loan_Status']  # y =target variable (approved or not)

#Step 9: Training aur Testing data split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 10: Model training 
model = RandomForestClassifier(random_state=42)  # RandomForest Classifier is used , best for classification purpose and friendly with nun values 
model.fit(X_train, y_train)  # Model training

#Step 11:  Prediction from model
y_pred = model.predict(X_test)

#Step 12: Model evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred)*100,"%")  # Accuracy check
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))  # Confusion matrix shows predictions
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Precision, Recall, F1-Score  
