The following steps are taken for creating the age detection software
* Import the necessary libraries 
* Identify the path to the dataset and read the dataset. The dataset was a big file that couldn't be included on this repository, but a copy of it can be found here https://www.kaggle.com/datasets/jangedoo/utkface-new. 
* Parse the name of each photo, where the first parameter is age, and the second parameter is gender.
* Transform the dataset to a data frame so it's easier to locate information.
* Classify female and male genders where 0 represents male and 1 is female. 
* Plot 25 images from the dataset and show the ratio of male to female. This is to verify the code is working before the CNN algorithm is trained with unprepared data, and the parsed information about age and gender are responding to images. 
* First normalize then standardize the size of images before they are sent for CNN analysis layers. 
* CNN layers of analysis, including the Flatten and Dense methods in the photo analysis layers. 
* Save the model in a file located in the same directory as ageDetection.py, and the model is then loaded from the saved file. This can avoid retraining the model every time the code is run, which is convenient. Notice, if you're running the python scrip file for the first time it will take a while until the model is trained. For the first time while the model trains it also gets saved in a file which will be used to make prediction at a faster time when you use the program again. 
* Provide the index of the photo you want to predict the age from.
* Currently the output of the program looks like the following lines
<br />  Original Age: 56
<br />  1/1 [==============================] - 0s 235ms/step
<br />  Predicted Age: 55
