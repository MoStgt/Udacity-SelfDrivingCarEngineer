# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./imagesWriteUp/ExploratoryTrainData.PNG "Training Data"
[image10]: ./imagesWriteUp/keepLeftGrayRGB.PNG "Gray vs RGB image"
[image11]: ./MyTrafficSigns/left_turn.png "Dangerous curve to the left"
[image12]: ./MyTrafficSigns/noCarAllowed.png "Vehicles over 3.5 metric tons prohibited"
[image13]: ./MyTrafficSigns/noOvertake.png "No passing"
[image14]: ./MyTrafficSigns/rightWay.png "Right of way"
[image15]: ./MyTrafficSigns/slippery.png "Slippery road"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MoStgt/Udacity-SelfDrivingCarEngineer/blob/master/Project3/CarND-Traffic-Sign-Classifier-Project-master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data set is structured. How many examples of each class is available in the training data set

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to achieve a better prediction nevertheless it was not enough to reach a prediction of 93%

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image10]

As a last step, I normalized the image data. This step led to a better prediction which resulted in a prediction over 93%


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input = 400 Output = 120        									|
| RELU					|												|
| Fully connected		| Input = 120 Output = 84       									|
| RELU					|												|
| Fully connected		| Input = 84 Output = 43       									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epochs of 70, a batch size of 128 and a learning rate of 0.00096

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.9%
* test set accuracy of 93.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I started with the original LeNet architecture which was introduced in the chapters.
* What were some problems with the initial architecture? I started with RGB images and when I switch to Grayscale images I needed to adjust the initial architecture from the input of a 32x32x3 image to a 32x32x1 image
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The architecture wasn't adjusted
* Which parameters were tuned? How were they adjusted and why? The learning rate and the epochs were tuned until the accuracy of the test set was satisfying. The learning rate was adjusted from initially 0.001 to 0.00096 in 0.0001 steps and the epochs were increased in steps of 10 epochs. I got the impression that the learning rate was too high so that the optimizer couldn't find the right values and due to a smaller learning rate I increased the epochs.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? A dropout layer can help to achieve higher accuracy on random images. It will reduce the probability of overfitting. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road    		| Slippery Road  									| 
| Right-of-way at the next intersection     			| Children crossing										|
| No passing				| No passing									|
| no vehicles     		| Vehicles over 3.5 metric tons prohibited					 				|
| Dangerous curve to the left			| Dangerous curve to the left     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Slippery Road  (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

(23, b'Slippery road'): 100.00%
(19, b'Dangerous curve to the left'): 0.00%
(10, b'No passing for vehicles over 3.5 metric tons'): 0.00%
(42, b'End of no passing by vehicles over 3.5 metric tons'): 0.00%
(21, b'Double curve'): 0.00%

(28, b'Children crossing'): 63.12%
(11, b'Right-of-way at the next intersection'): 32.57%
(30, b'Beware of ice/snow'): 4.04%
(27, b'Pedestrians'): 0.27%
(34, b'Turn left ahead'): 0.00%

(9, b'No passing'): 100.00%
(41, b'End of no passing'): 0.00%
(12, b'Priority road'): 0.00%
(40, b'Roundabout mandatory'): 0.00%
(0, b'Speed limit (20km/h)'): 0.00%

(16, b'Vehicles over 3.5 metric tons prohibited'): 100.00%
(42, b'End of no passing by vehicles over 3.5 metric tons'): 0.00%
(10, b'No passing for vehicles over 3.5 metric tons'): 0.00%
(35, b'Ahead only'): 0.00%
(0, b'Speed limit (20km/h)'): 0.00%

(19, b'Dangerous curve to the left'): 100.00%
(35, b'Ahead only'): 0.00%
(23, b'Slippery road'): 0.00%
(34, b'Turn left ahead'): 0.00%
(20, b'Dangerous curve to the right'): 0.00%

 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


