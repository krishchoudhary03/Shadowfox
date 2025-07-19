# Step 1: Imported required libraries for CNN development 
import tensorflow as tf # As known tensorflow is imp for Ai-Ml development , provides library for models
from tensorflow.keras import layers, models#Used for CNN layers and Model development 
from tensorflow.keras.preprocessing.image import ImageDataGenerator#Used for img loading and augmentation{like-Rotation,Fliping}
import numpy as np # for array operations
import matplotlib.pyplot as plt # for img and graphs representation
import os#for file path operations

# Step 2: Data Preprocessing and Augmentation 
train_gen = ImageDataGenerator(
    rescale=1./255,#convert pixel value to 0-1 range 
    rotation_range=30,#img roataion for marking diversity in data
    zoom_range=0.2,#random zoom in - out
    horizontal_flip=True #random horizontal fliping of img 
)
val_gen = ImageDataGenerator(rescale=1./255)#rescale data for validation purpose
test_gen = ImageDataGenerator(rescale=1./255)#rescale data for testing purpose

# Step 3: Loading  data from local file folders
train_data = train_gen.flow_from_directory(
    r"C:\Users\ASUS\Desktop\shadowfox\train_set",#folder of training data
    target_size=(128, 128),#resize img to 128*128
    batch_size=32,#train images in 32-32 limit batches
    class_mode='categorical'#One-hot encoding of labels (multi-class)
)

val_data = val_gen.flow_from_directory(
    r'C:\Users\ASUS\Desktop\shadowfox\val_set',#folder of validating data
    target_size=(128, 128),#same as  training  data
    batch_size=32,#same as  training data
    class_mode='categorical'#same as  training data
)

test_data = test_gen.flow_from_directory(
    r'C:\Users\ASUS\Desktop\shadowfox\test_set',#folder of testing data
    target_size=(128, 128),#same as  training  data
    batch_size=32,#same as  training  data
    class_mode='categorical',#same as  training  data
    shuffle=False# shuffle should be off while tesing ,else it will disturnb sequential process
)#location of all 3 folders should be same in location .

# Step 4: CNN Model development
model = models.Sequential([#adding  layers step by step
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),#1st Conv layer, 32 filters
    layers.MaxPooling2D(2, 2),# 1st Pooling layer - reduce size

    layers.Conv2D(64, (3, 3), activation='relu'),# 2nd Conv layer
    layers.MaxPooling2D(2, 2),# 2nd Pooling

    layers.Conv2D(128, (3, 3), activation='relu'),#3rd Conv layer
    layers.MaxPooling2D(2, 2),#3Rd pooling

    layers.Flatten(),# 2D to 1D flattening-{conversion}
    layers.Dense(128, activation='relu'),# Dense (Fully connected) layer
    layers.Dense(train_data.num_classes, activation='softmax')# Final output (no. of classes)
])

# Step 5: Compile Model- seting rules for training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#used Adam optimizer for fast and adaptive responsing 
# Step 6: Train the Model
history = model.fit(
    train_data,# Training images
    validation_data=val_data,# Validation images
    epochs=15  # increase epochs for better accuracy, as it is no of time data gonna train
)

# Step 7: Evaluate Model
test_loss, test_acc = model.evaluate(test_data)#checking loss and accuracy at testing
print(f" Test Accuracy: {test_acc * 100:.2f}%")

# Step 8: Save Model
model.save("my_tagging_model.h5")#saving model in file format 
print(" Model saved as 'my_tagging_model.h5'")
