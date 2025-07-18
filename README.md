# Shadowfox
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

