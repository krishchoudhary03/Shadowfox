# Shadowfox {task 1}
CNN-Based Cats vs Dogs Classifier

This is a beginner-level AI/ML project developed during the **ShadowFox Virtual Internship**.  
The goal of this project is to **train a Convolutional Neural Network (CNN)** to classify whether an image is of a **cat** or a **dog**.

 

  Project Summary

- **Model Type:** CNN (Convolutional Neural Network)
- **Framework Used:** TensorFlow with Keras
- **Task:** Binary image classification (Cats 🐱 vs Dogs 🐶)
- **Data Format:** Images (JPEG/PNG)
- **Model Output:** Predicts whether the given image is a cat or a dog, along with confidence %

 
  Dataset Details

The dataset is organized into three main folders:

- `train/` → Used to train the model  
- `validation/` → Used to validate the model during training  
- `test/` → Used for final testing and predictions  

  **Total Images:** 280+  
- Training: ~ 200 images  
- Validation: ~ 40 images  
- Testing: ~ 40 images  
Each category (cats/dogs) has around 50% distribution.

 

  Model Functionality

- **Input:** Image of any dog or cat (can be uploaded locally or provided via a URL)
- **Output:** The model returns:
  - Predicted label (`cat` or `dog`)
  - Confidence percentage
  - Optionally, the image is displayed with prediction title

 

  How It Works

1. **Data Preprocessing**
   - Images resized to **128x128 pixels**
   - Normalization applied (pixel values scaled between 0 and 1)
   - Data Augmentation (optional)

2. **Model Architecture**
   - Multiple convolution layers (for feature extraction)
   - MaxPooling layers (to reduce spatial dimensions)
   - Fully connected (Dense) layers
   - Output layer with `softmax` activation (2 classes)

3. **Training**
   - Loss Function: `categorical_crossentropy`
   - Optimizer: `Adam`
   - Metrics: `accuracy`
   - **Epochs:** Usually trained for 10 epochs (depends on dataset size & hardware), I did it on 15 epochs
   - **Training Time:** Varies — approx. **2-3 minutes** on mid-level PCs for 200 images

 

Prediction Interface

- The trained model (`my_image_model.h5`) is loaded using Keras.
- Users can either:
  - Paste an online image URL
  - Use an image stored locally on their PC
- The image is preprocessed in the same way as training images (resized, normalized).
- The model then outputs the predicted class and its confidence.

 

  Results

- Accuracy after training: **~85%–95%** depending on data split and training time.
- Misclassification can happen if:
  - Image quality is poor
  - Unrelated images are used (like lions or other animals)
 

  Future Improvements

- Add more categories (multi-class classification)
- Improve accuracy by using transfer learning (e.g., MobileNet, ResNet)
- Deploy as a web app using Flask/Streamlit
Author

Developed by **Krish Choudhary**  
Under the **ShadowFox AIML Internship Program**  
Connect on  www.linkedin.com/in/krish-choudhary-55b9a030b | #ShadowFoxIntern | #AIML | #CNNModel








#Shadowfox-Task-2
loan-approval-prediction
Project Title
Loan Approval Prediction using Machine Learning (Random Forest Classifier)

Project Objective
The goal of this project is to predict whether a loan application will be approved based on applicant details such as income, credit history, employment status, and more. The model used for this prediction is Random Forest Classifier.

This project is divided into two parts:

L_P_M-Base.py — Training the model on historical data.
prediction status.py — Taking manual input and using the model to predict loan approval.
Internship Context
This project was completed as part of the ShadowFox Virtual Internship Program under the AIML (Artificial Intelligence and Machine Learning) track. It is one of the intermediate-level tasks designed to help interns understand real-world applications of machine learning, especially in predictive modeling and classification problems.

The task specifically focuses on using ML to build a loan prediction system, which helps banks or financial institutions decide whether a loan should be approved for an applicant or not.

How the Project Works
1. L_P_M-Base.py
This script:

Loads the dataset (loan_prediction.csv)
Handles missing values
Encodes categorical columns (like Gender, Married, etc.) into numbers
Trains a RandomForestClassifier on the data
Saves the feature names into a file called trained_model_data.py
This file prepares everything needed for manual prediction later.

2. prediction status.py
This script:

Loads the dataset again
Cleans and encodes it just like the training script
Re-trains the same model on the full dataset (for simplicity)
Takes input from the user via the terminal (like gender, income, loan amount, etc.)
Formats the input to match training data format
Predicts whether the loan will be approved or not
How to Run the Code
Step 1: Train the model
Run the L_P_M-Base.py file in your terminal:

Python L_P_M-Base.py

Step 2: Make a prediction
Run the prediction status.py file in your terminal:

python prediction status.py

You will be asked to enter details manually in the terminal. Based on your input, the model will give you a prediction.

Dataset
The dataset loan_prediction.csv should be placed in the same folder as the two .py files. It includes:

Gender
Marital Status
Number of Dependents
Education
Employment Status
Applicant Income
Coapplicant Income
Loan Amount and Term
Credit History
Property Area
Tools Used
Python
pandas
scikit-learn
Project Outcome
You’ll have a simple command-line tool where users can enter loan application details and receive a prediction result about loan approval.

This project is designed to help beginners understand the full machine learning workflow: from data preprocessing and model training to manual prediction.

Credits
Developed by Krish Choudhary Under the ShadowFox AIML Internship Program Connect on www.linkedin.com/in/krish-choudhary-55b9a030b | #ShadowFoxIntern | #AIML | #ClassificationModel |#RandomForestClassifierModel

