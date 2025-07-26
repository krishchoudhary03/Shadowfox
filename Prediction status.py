# Importing required libraries
import numpy as np                      # For numerical operations
import pandas as pd                    # For handling data in tabular form (dataframes)
from sklearn.preprocessing import LabelEncoder  # To convert text labels to numeric values
from sklearn.ensemble import RandomForestClassifier  # Machine learning model
from sklearn.model_selection import train_test_split  # To split data for training and testing

# Load the dataset from a CSV file
data = pd.read_csv("loan_prediction.csv")  # Reading the CSV file into a Dataframe

# Fill missing values in categorical and numerical col 
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])  # Fill missing Gender with most common value
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])   
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])   
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])   
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])   
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])  # Fill missing Term
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())  # Fill missing Loanamount with median

# Drop useless column  
data.drop('Loan_ID', axis=1, inplace=True)

# Convert categorical columns into numeric using Labelencoder
le = LabelEncoder()  # Create an object of Labelencoder
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols:
    data[col] = le.fit_transform(data[col])  # Convert text labels to numbers for all

# Replace '3+' with 3 in Dependents and convert it to integer
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].astype(int)  

# Separate features and target variable
X = data.drop('Loan_Status', axis=1)  # X = all columns  
y = data['Loan_Status']  # y = Loanstatus column  

# Create and train the Random Forest model
model = RandomForestClassifier(random_state=42)  # Create the classifier with a fixed random state for reproducibility
model.fit(X, y)  # Training the model using the data

# Ask user to enter application
print("Enter your loan application details:")

# Take all required inputs from user 
gender = int(input("Gender (Male = 1, Female = 0): "))
married = int(input("Married? (Yes = 1, No = 0): "))
dependents = int(input("Dependents (0 / 1 / 2 / 3): "))
education = int(input("Education (Graduate = 0, Not Graduate = 1): "))
self_employed = int(input("Self Employed? (Yes = 1, No = 0): "))
applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_amount_term = float(input("Loan Amount Term (in days): "))
credit_history = float(input("Credit History (1.0 = Good, 0.0 = Bad): "))
property_area = int(input("Property Area (Urban = 2, Semiurban = 1, Rural = 0): "))

# Dictonary for all input as wanted format 
input_dict = {
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
}

# Convert the dictionary to a DataFrame so the model can use it
input_df = pd.DataFrame(input_dict)

# Make prediction using the trained model
prediction = model.predict(input_df)

# Print result based on prediction
if prediction[0] == 1:
    print("\n Loan Approved!")  # If model predicted 1, loan is approved
else:
    print("\n Loan Not Approved.")  # If model predicted 0, loan is not approved
