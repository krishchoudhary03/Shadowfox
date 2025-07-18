# Step 1: Required Libraries
import tensorflow as tf #tensorflow for ml tasks
from tensorflow.keras.models import load_model #loading tarined model
from tensorflow.keras.preprocessing import image#image loading and conversion
import numpy as np #for array operations
import matplotlib.pyplot as plt# for img display
import requests#image file handling
from PIL import Image#handling byte-stream 
from io import BytesIO#handling path 

# Step 2: Load Trained Model
model = load_model("my_tagging_model.h5")#loading trained model
print("✅ Model Loaded Successfully!")

# Step 3: Define Class Labels (based on training)
class_labels = ['cats', 'dogs'] # model will answer either of these option 

# Step 4: Paste your img URL here
img_url ="https://pm1.narvii.com/7600/70da7db18af73fb0c2fdde9729299c4ffe5fcf17r1-1100-825v2_hq.jpg"#paste url here

# Step 5: Image Loading & Preprocessing
response = requests.get(img_url)#loading image from url
img = Image.open(BytesIO(response.content)).convert('RGB')#open and converting to RGB
img = img.resize((128, 128))# Resize as per model input
img_array = image.img_to_array(img) / 255.0#mormalizing
img_array = np.expand_dims(img_array, axis=0)#arranging for model {shape}

# Step 6: Prediction
prediction = model.predict(img_array)#taking image prediction
predicted_class = np.argmax(prediction[0])#higher most porbability counting 
confidence = prediction[0][predicted_class] * 100#confidence percentage 

# Step 7: Output Result
print(f"Prediction: {class_labels[predicted_class]}")#answer of prediction
print(f"Confidence: {confidence:.2f}%")# percent of confidence / accuracy

# Step 8: Show Image with prediction percent
plt.imshow(img)
plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
