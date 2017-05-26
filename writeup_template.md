**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Exploratory_visualization.png "Dataset Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/1.jpg "Traffic Sign 1"
[image5]: ./test_images/2.jpg "Traffic Sign 2"
[image6]: ./test_images/3.jpg "Traffic Sign 3"
[image7]: ./test_images/4.jpg "Traffic Sign 4"
[image8]: ./test_images/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/VikasPatil-GitHub/CarND-Traffic-Sign-Classifier.git)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of training samples for each traffic sign class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the prediction accuracy of the LeNet architecture was better on grayscale images than on images in RGB format. The reason being the CNNs tend to pick color blobs as features in an RGB image which can be prevented by using a grayscale image.

As a last step, I normalized the image data to have zero mean and equal variance because adding very small values to very large values can introduce lot of errors and a badly conditioned data makes the optimization process harder, thus increasing the training time. Whereas a well conditioned data make the optimization process easier and faster thus decreasing the training time.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 data |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6				|
| Fully connected		| Input = 400 Output = 120	|
| Dropout		| Probability = 0.75	|
| Fully connected		| Input = 120 Output = 84	|
| Dropout		| Probability = 0.75	|
| Fully connected		| Input = 84 Output = 43	|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a batch size of 128 and trained the model for 25 epochs. I used a learning rate of 0.001 with keep probability of 0.75 for the dropout layer in fully connected layers.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4 %
* validation set accuracy of 93.7 % 
* test set accuracy of 91.4 %

If a well known architecture was chosen:
* What architecture was chosen?

LeNet architecture was chosen for the project.

* Why did you believe it would be relevant to the traffic sign application?

As part of the LeNet lab to classify the characters and numbers LeNet architecture performed well with a prdiction accuracy of 98% on grayscale images. Hence the LeNet architecture was chosen for traffic sign classifier application. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The model's accuracy was 99.6% on the training dataset and 93.2% and 92.2% on the validation and test dataset. This proves that the LeNet model is suitable for the traffic sign calssification / prediction.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      		| Double curve   									| 
| Keep left     			| Keep left 										|
| Roundabout mandatory					| Roundabout mandatory											|
| Children crossing	      		| Children crossing					 				|
| Speed limit (20km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I used a different strategy here for testing the model on the images downloaded from the web. Instead of using more than 5 images and getting the top five predictions, I used only 5 images to check the robustness of model. The model was able to predict 4 out of the 5 signs giving a accuracy of 80%. This accuracy is achieved by using only two data preprocessing techniques viz. grayscale conversion and normalization. I believe the acuuracy can be further increased by using augmentating the dataset and adding more samples for the traffic sign classes with fewer samples. 

For the first image, the model is relatively sure that this is a double curve (probability of 0.6).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59         			| Double curve   									| 
| 1.0     				| Keep left 										|
| .96					| Roundabout mandatory											|
| .84	      			| Children crossing					 				|
| .000022, 0.99			    | Speed limit (20km/h), Speed limit (30km/h)      							|


For the second, third and foruth images the probabilities were close to 1 indicating accurate prediction. Whereas for the fifth image the model was leaning towards 30km/h rather than 20km/h. The reason for this migth be very less samples for 20km/h speed limit sign as seen in the dataset visualization. This can rectified by adding more samples for the 20km/h speed limit sign.



