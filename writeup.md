#**Traffic Sign Recognition** 

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

[image1]: ./examples/dataset_hist.png "Visualization"
[image2]: ./examples/sign_nclasses.png "nclasses"
[image3]: ./examples/train_valid.png "Train"
[image4]: ./examples/0.png "Traffic Sign 1"
[image5]: ./examples/17.png "Traffic Sign 2"
[image6]: ./examples/34.png "Traffic Sign 3"
[image7]: ./examples/39.png "Traffic Sign 4"
[image8]: ./examples/40.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/horagong/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cell of the IPython notebook.

As a first step, I tried to convert images through following methods, like grayscale, histogram equalizing, normalizaing.

But the result of accuracy had not big difference.
So I just do scaling because the value of train dataset was (0, 255) and it is different with the value of png which is read by matplotlib.image


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the already seperated dataset in the first cell. I looked at the distribution of the dataset. I could see that the validation data and test data distribution are the same as train data by random sampling. Some traffic signs has relatively smaller number than the others. But I thought those classes are not very rare and so the dataset is not very sckewed.

At first I tried to generate some data to balance it by data_augment method in the fourth cell. But the result of accuracy was not big different as I guessed. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x70 	|
| RELU					|												|
| Max pooling 2x2    	| 2x2 stride, outputs 15x15x70 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x70 	|
| RELU                  |                                               |
| Max Pooling 2x2       | 2x2 stride, output 6x6x70                     |
| Fully connected		| 1000       									|
| RELU                  |                                               |
| Dropout               | keep_prob 0.5                                 |
| Fully connected       | 1000                                          |
| RELU                  |                                               |
| Dropout               | keep_prob 0.5                                 |
| Softmax				|       							    		|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used a graph of train and validation accuracy. I stopped training epoch when it seems to be saturated. When it seems to increase slowly, I grew the learning rate.

![alt text][image3] 


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.946
* test set accuracy of 0.899

I used LeNet because it's well known and relatively simple network for image recognition. At first it didn't have an accuracy over 0.93. So I increased the number of filter depth and fully connected neurons.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The test images might not be difficult to classify because it has the same quality as the dataset.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| #sign 0      	    	| #sign 0   									| 
| #sign 17     			| #sign 17										|
| #sign 34				| #sign 34										|
| #sign 39	      		| #sign 39	    				 				|
| #sign 40  			| #sign 40            							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

All prediction of the test image was probability 1.