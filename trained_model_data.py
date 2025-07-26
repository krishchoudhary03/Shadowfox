from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Pre-trained model instance (random forest)
model = RandomForestClassifier()
feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
