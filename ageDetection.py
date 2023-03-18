# https://www.hackersrealm.net/post/gender-and-age-prediction-using-python
# Import the necessary modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore') 
#%matplotlib inline
import tensorflow as tf
from tensorflow.keras.utils import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from PIL import Image

# Store image paths, get age and gender values. 
image_paths = []
ages_dictionary = []
gender_types = []

# Tqdm is often used for reading data, processing, training machine learning convolution_models 
# In this section tqdm is used for reading the dataset
for file in tqdm(os.listdir("./UTKFace/")):
    #image_path = os.path.join("./UTKFace/", file)
    temp = file.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(os.path.join("./UTKFace/", file))
    ages_dictionary.append(age)
    gender_types.append(gender)  

# Put data in a data structure/frame.
datastructure = pd.DataFrame()
datastructure['image'], datastructure['age'], datastructure['gender'] = image_paths, ages_dictionary, gender_types
#datastructure.head()

# Assign gender to gender values. Uncomment following lines to show image. 
gender_types_dict = {1:'Male', 0:'Female'}

# Statistical distribution of Age and Gender in the dataset using Seaborn visualization
sns.distplot(datastructure['age'])
sns.countplot(datastructure['gender'])

# To display grid of images_dictionary
plt.figure(figsize=(20, 20))
# Locate random 25 images_dictionary from the dataset
files = datastructure.iloc[0:25]

# Using itertuples() to iterate over photos and return each column as tuple object
# Tuples method of data lookup is used for time effeciency and data security 
for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_types[gender]}")
    plt.axis('off')

# In this section tqdm is used for reading images_dictionary, and later extracting features of images_dictionary
def extract_features(images_dictionary):
    features = []
    for image in tqdm(images_dictionary):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)  
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features
image = extract_features(datastructure['image'])
image.shape

# Normalize the image
image = image/255.0

# Convert Age and Gender into numpy
photo_gender = np.array(datastructure['gender'])
photo_age = np.array(datastructure['age'])
input_size = (128, 128, 1)

# Create convolution_models
inputs = Input((input_size))
# Convolutional layers
convolution_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxpooling_1 = MaxPooling2D(pool_size=(2, 2)) (convolution_1)
convolution_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxpooling_1)
maxpooling_2 = MaxPooling2D(pool_size=(2, 2)) (convolution_2)
convolution_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxpooling_2)
maxpooling_3 = MaxPooling2D(pool_size=(2, 2)) (convolution_3)
convolution_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxpooling_3)
maxpooling_4 = MaxPooling2D(pool_size=(2, 2)) (convolution_4)

# The above convolution layers are in form of matrics meaning the data is two dimensional
# Faltten method is used for convering matrics data to vectors which is one dimensional
flatten = Flatten() (maxpooling_4)

# Fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)
dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)
convolution_model = Model(inputs=[inputs], outputs=[output_1, output_2])
convolution_model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

if os.path.exists('/eccs/home/daqasimi18/CS488/saved_trained_convolution_model.h5') == True:
    trained_convolution_model = tf.keras.models.load_model("saved_trained_convolution_model.h5")
    image_index = 100
    print("Original Age:", photo_age[image_index])
    # predict from convolution_model
    pred = trained_convolution_model.predict(image[image_index].reshape(1, 128, 128, 1))
    #pred_gender = gender_types_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print("Predicted Age:", pred_age)
    plt.axis('off')
    plt.imshow(image[image_index].reshape(128, 128), cmap='gray');
else:
    # In the training section of the convolution_model thirty percent of the dataset is used for validation or 
    # testing the accuracy of convolution_model batch_size is for the number training examples in one iteration,   
    # and epoch is one complete pass of the training dataset through the algorithm 
    train_model = convolution_model.fit(x=image, y=[photo_gender, photo_age], batch_size=32, epochs=30, validation_split=0.3)

    # Save the convolution_model
    convolution_model.save("saved_trained_convolution_model.h5")
    trained_convolution_model = tf.keras.models.load_model("saved_trained_convolution_model.h5")
    
    acc = train_model.history['gender_out_accuracy']
    epochs = range(len(acc))

    # plot results for age
    loss = train_model.history['age_out_loss']
    val_loss = train_model.history['val_age_out_loss']
    epochs = range(len(loss))

    #plt.plot(epochs, loss, 'b', label='Training Loss')
    #plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    #plt.title('Loss Graph')
    #plt.legend()
    #plt.show()
    image_index = 100
    print("Original Age:", photo_age[image_index])
    # predict from convolution_model
    pred = trained_convolution_model.predict(image[image_index].reshape(1, 128, 128, 1))
    #pred_gender = gender_types_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print("Predicted Age:", pred_age)
    plt.axis('off')
    plt.imshow(image[image_index].reshape(128, 128), cmap='gray');