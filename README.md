## Project: Build a Traffic Sign Recognition Program

### Overview

In this project, I will use deep neural networks and convolutional neural networks to classify traffic signs. I will then trained a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then test my model program on new images of traffic signs that I find on the web.

Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES.


[//]: # (Image References)
[image1]: ./images/Visualize-German-Traffic.png "Visualize-German-Traffic"
[image2]: ./images/Distribution-of -Labels-Per-Class.png "Distribution-of -Labels-Per-Class"
[image3]: ./images/final_architecture.png "final_architecture"
[image4]: ./images/test_image.png "test_image"
[image5]: ./images/predicted_image.png "predicted_image"
 

# Visualize the German Traffic Signs Dataset

The traffic signs examples are not uniformly distributed across class labels.


Shown below is example of German Traffic Signs image data set.

![alt text][image1]

# Distribution of Labels Per Class

![alt text][image2]


# Data preprocessing

I converted all images to grayscale to make edges more distinguishable and also to get computational efficiency in subsequent operations.I also performed histogram equalization on the resulting grayscaled image to equalize the most frequent intensity values so as to get better contrast enhancement.I standardized data by subtracting mean from image array then dividing by its standard deviation.By subtracting mean, we are eliminating the influence of mean as large mean value can cause accuracy loss.


# Set up the training, validation and testing data for my model

The dataset appears imbalanced across the labels.For examaple,'label 5' has 1800,'lable 6' has 400 and 'label 32' has 800 examples.So I have chosen to use train and test split technique in a stratified fashion in an attempt to deal this imbalanced class distibution.
I have split the original training dataset into 67% training dataset to train the model and to evaluate and perform validation accuracy against 33% validation dataset.I have used a seed value of 7 for the purpose of reproductibily of that split so that it doesnt effect evaluation outcome while I tune other hyperparameters.I have also shuffled the training data set.
The traffic signs examples are not uniformly distributed across class labels.Generating addional data may help make the model train better and subsequently in gaining a better test accuracy.My preprocessing steps do not include data augmentation at this stage.


# Final architecture 

My final architecture implemets the LeNet-5 architecture,a pioneering convolutional network by Yann LeCun.LeNet-5 is desgined to work on small fixed size input and takes 32x32 input data.Since our traffic sign images are 32x32x3 pixels,the LeNet-5 soultion will be a good fit for this classification problem.
My implementation slighly differs from the original boilerplate solution as I have used tuned hyper parameters,optimizer,regularisation and feaure maps that I have found work better for this given problem in the environement where I performed the experimeation and tested my model.
The input is going to be a 4-D tensor input - the number of samples,number of color channels,width(32) and height(32).Kernal is also 4-D convolutional filter - feature maps x number of color channels x width x height.
The architecture includes convolutional layers followed by a RELU activation and max pooling layer which is used to reduce the size of features.This structure is repeated twice.Then I flatten the features into a vector and put them into fully connected layers that are stacked at the output end.I have used dropout to prevent overfitting only during training between fully connected layers.The output here that we get is a set of 'logits'.We convert the outputs to a probability distribution using the softmax function.The softmax normalizes the unscaled logit outputs.
Below summarizes the final network architecture that I have used in the model.


![alt text][image3]

## Layer 1: Convolutional.
The first Convolutional layer takes original image as input.It applies 5x5 kernal,stride of length 1,depth of 6 with VALID padding.This result in a feature map of 28x28x6 output values.
Activation layer uses RELU activation function.
Pooling layer taking the max over 2x2 patches.This result in output shape of 14x14x6.

## Layer 2: Convolutional.
The input is 14x14x6 that resulted from the first Convolutional layer's max pooling operation. This layer applies 5x5 filter,stride of length 1,a depth of 32 with VAILD padding.This result in a feature map of 10x10x32. 
Activation layer uses RELU activation function.
Pooling layer taking the max over 2x2 patches.This result in output shape of 5x5x32.
Flatten layer flattens the inputs to 2D (batches,length) to be used by fully connected layers.The batch size is ignored here as it remains unaltered.After multiplying the output shape of max pooling layer 5x5x32 results 800 neurons.

## Layer 3: Fully Connected.
Fully connected layer with 800 neurons with RELU activation 
Dropout layer with dropout probaility 0.5

## Layer 4: Fully Connected.
Fully connected layer with 400 neurons with rectifier activation.
Dropout layer with dropout probaility 0.5

## Layer 5: Fully Connected (Logits).
Fully connected layer with 120 neurons and rectifier activation.
Output :
This should have 43 outputs.


# Train model - optimizer, batch size, epochs, hyperparameters

I have optimized the model with Momentum Optimizer.I have chosen tf.train.exponential_decay function for learning rate as that controls the decay rate.I also evaluated the model with the adam optimizer but noticed a better result with momentum optimizer.
For weights,I have used normal distribution function that truncates distribution from being too extreme by binding the value between truncation range.I tried letting the moded trianed with very low inititializing value of mean 1e-4 and Standard deviation of 3e-5,after running the model with a batch size of 128 for 70 epochs,the validation accuracy was found 94.8% and test accuracy 88.8%. Whereas with a much lower epochs run,I was able to get a better validaiton accuracy of 96.6% and test accuracy of 90.9% with a mean 0(zero) and standard deviation 1e-2 range. So I have chosen the mean 0 and standard deviation 1e-2 for weight initialisation.
I have used the softmax crossentropy loss function to evaluate the weights.This function searches through different weights for the network.
As this is a multiclass classification problem,I used one hot encoding of the class values.It transformed the vector of class integers into a binary matrix. I have trained the network for 80 epochs and validated against 33% of the training data.


# Approach I took in coming up with a solution to this problem
Since the LeNet-5 works well on 32x32 input data, and our traffic sign datasets contains 32x32x3 image data,I decieded to implement this well known solution as I found it is a plug and play solution with minor tweak require for hyperparameters.
While training the model,first,I evaluated my model against the subsets of training data which is known as the validation datasets that I created during preprocesisng steps.This was done to make sure to check if the model was overfitting by looking at the resulted accuracy metric from the trained and validation datasets.After that, I performed the evalution of my model on the actual test dataset.
As this is a classification problem,I let the model run and report classification accuracy as the metric. I let the model trained with backpropagation based optimization algorithms for 40~80 epochs.I noticed the validation accuracy increase with an increase in number of epochs, but the rate of improvement was very small.I tuned the hyperparameters.I captured the results in a log and tuned the parameters as I was training with different hyperparameters values and worked on improved version of the model.At the end I chose a model and values for hyperparameters that I found gave the best results in the the environement where I performed the tests.After 70 epochs,I got a validation accuracy of 97.1% and a test accuracy of 91.0%. Then I increased the number of epochs to 80, and found a slight improvement of the validation accuracy to 97.5% and test accuracy to 91.7%.I didn't want to overtrain the model which can cause overfitting issue so I stopped training the model further.
 
 
 
# Five candidate images of traffic signs 
 
I have chosen six traffic sign images from web.After looking at the original RGB images and chosen new images,I think, the shapes,edges,and intensities of pixels of new images will make difficult for the trained model to make correct predictions some of these unseen images.I have plotted the images below to help me understand the qualities and differences of the images. 
 
# Test images

![alt text][image4]
 
 
# Predicted images

![alt text][image5]
 
 