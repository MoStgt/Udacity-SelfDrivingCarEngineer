# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Suggested NVIDIA model was taken with some small adaption. The model takes as an input an image size of (160, 320, 3) instead of originally (200, 66, 3) in addition to that a drop out layer was introduced to avoid overfitting which led to a better validation result


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 78). The initial value of the dropout layer was 0.5 which led to satisfying results.

The model was trained and validated on the provided data set. Through the split of the data in training and validation set it is ensured that the model was not overfitting (code line 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the provided data set, a combination of center lane driving, recovering from the left and right sides of the road.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first start with already known models.

My first step was to use a convolution neural network model similar to the NVIDIA I thought this model might be appropriate because it is introduced as "End to End Learning for Self-Driving Cars" by NVIDIA.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. Nevertheless when it was tested in autonomous mode it didn't show the expected behavior that's why a dropout layer was introduced to avoid overfitting. The dropout rate was set to 0.5

The final step was to run the simulator to see how well the car was driving around track one. The result was good and satisfying.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-83) consisted of a convolution neural network with the following layers and layer sizes.
O = (W-K+2P)/S + 1
 $\frac{\frac{x}{1}}{x - y}$
where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.

Layer 1: Convolution with strides=(2,2), kernel 5x5 and output shape 31x98x24, with elu activation

Layer 2: Convolution with strides=(2,2), valid padding, kernel 5x5 and output shape 14x47x36, with elu activation and dropout

Layer 3: Convolution with strides=(2,2), valid padding, kernel 5x5 and output shape 5x22x48, with elu activation and dropout

Layer 4: Convolution with strides=(1,1), valid padding, kernel 3x3 and output shape 3x20x64, with elu activation and dropout

Layer 5: Convolution with strides=(1,1), valid padding, kernel 3x3 and output shape 1x18x64, with elu activation and dropout

flatten 1152 output

Layer 6: Fully Connected with 100 outputs and dropout

Layer 7: Fully Connected with 50 outputs and dropout

Layer 8: Fully Connected with 10 outputs and dropout

dropout was set aggressively on each layer at .25 to avoid overtraining ###Output

Layer Fully Connected with 1 output value for the steering angle.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
