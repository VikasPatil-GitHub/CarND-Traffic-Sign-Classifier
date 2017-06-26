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

---
### Writeup / README

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of training samples for each traffic sign class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of the image data

As a first step, I decided to convert the images to grayscale because the prediction accuracy of the LeNet architecture was better on grayscale images than on images in RGB format. The reason being the CNNs tend to pick color blobs as features in an RGB image which can be prevented by using a grayscale image.

As a last step, I normalized the image data to have zero mean and equal variance because adding very small values to very large values can introduce lot of errors and a badly conditioned data makes the optimization process harder, thus increasing the training time. Whereas a well conditioned data make the optimization process easier and faster thus decreasing the training time.


#### 2. Final model architecture

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
 


#### 3. Training the model

To train the model, I used an Adam optimizer with a batch size of 128 and trained the model for 25 epochs. I used a learning rate of 0.001 with keep probability of 0.75 for the dropout layer in fully connected layers.

#### 4. Approach taken for finding a solution to get a good validation set accuracy

My final model results were:
* training set accuracy of 99.4 %
* validation set accuracy of 93.7 % 
* test set accuracy of 91.4 %

LeNet architecture was chosen for the project. As part of the LeNet lab to classify the characters and numbers LeNet architecture performed well with a prdiction accuracy of 98% on grayscale images. Hence the LeNet architecture was chosen for traffic sign classifier application. 

The model's accuracy was 99.6% on the training dataset and 93.2% and 92.2% on the validation and test dataset respectively. This proves that the LeNet model is suitable for the traffic sign calssification / prediction.

### Test the Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Model's predictions on new traffic signs and comparing the results to prediction on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      		| Double curve   									| 
| Keep left     			| Keep left 										|
| Roundabout mandatory					| Roundabout mandatory											|
| Children crossing	      		| Children crossing					 				|
| Speed limit (20km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

#### 3. Softmax probabilities for each prediction

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
