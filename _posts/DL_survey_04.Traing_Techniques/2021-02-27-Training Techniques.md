---
title:  "Deep Learning survey_6.Advanced Training Techniques"
search: true
categories:
  - Deep learning
date: February 27, 2021
summary: This post is the sixth part (Advanced Training Techniques) of summary of a survey paper.
toc: true
toc_sticky: true
header:
  teaser: /assets/images/thumbnails/thumb_basic.jpg
tags:
  - Deep Learning
last_modified_at: 2021-03-24T08:06:00-05:00
---


This post is the sixth part (Advanced Training Techniques) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
The main objective of this work is to provide an overall idea on deep learning and its related fields.

<br>

The advanced training techniques or components which need to be considered carefully for `efficient training of DL` approach. The techniques include `input pre-processing`, a better `initialization method`, `batch normalization`, alternative `convolutional approaches`, advanced `activation functions`, alternative `pooling techniques`, network `regularization` approaches, and better `optimization` method for training.  
This following sections are discussed on individual advanced training techniques individually.  

### 4.1. Preparing Dataset  

Presently different approaches have been applied before feeding the data to the network. The different operations to prepare a dataset are as follows; `sample rescaling`, `mean subtraction`, `random cropping`, `flipping data` with respect to the horizon or vertical axis, `color jittering`, `PCA/ZCA`, `whitening` and many more.  

<br>

### 4.2. Network Initialization  

*The initialization of deep networks has a big impact on the overall recognition accuracy*. For complex tasks with high dimensionality data training, a DNN becomes difficult because `weights should not be symmetrical` due to the back-propagation process. Therefore, effective initialization techniques are important for training this type of DNN.  
- Xavier initialization approach  
- LSUV (Layer-sequential unit-invariance)  
- He initialization: The distribution of the weights of *l<sup>th</sup>* layer will be a normal distribution with mean zero and variance *2/n<sub>l</sub>* which can be expressed as follows:  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation1.png" style="width:15%">
</p>

<br>

### 4.3. Batch Normalization  

Training deep neural networks with many layers is challenging as they can be sensitive to the initial random weights and configuration of the learning algorithm. One possible reason for this difficulty is `all layers are changed during an update`, *the update procedure is forever chasing a moving target*. For example, the weights of a layer are updated given an expectation that the prior layer outputs values with a given distribution. This distribution is likely changed after the weights of the prior layer are updated. This change in the distribution of inputs to layers in the network is referred to the technical name `'internal covariate shift'`.

Batch normalization is a technique for training very deep neural networks that `standardizes the inputs to a layer for each mini-batch`. This has the effect of `stabilizing` the learning process and dramatically `reducing the number of training epochs` required to train deep networks. It scales the output of the layer, specifically by standardizing the activations of each input variable per mini-batch, such as the activations of a node from the previous layer. Recall that standardization refers to rescaling data to have a mean of zero and a standard deviation of one, e.g. a standard Gaussian. This process is also called `'whitening'` when applied to images in computer vision.  

Standardizing the activations of the prior layer means that assumptions *the subsequent layer makes about the spread and distribution of inputs during the weight update will not change, at least not dramatically.* This has the effect of stabilizing and speeding up the training process of deep neural networks. Normalizing the inputs to the layer has an effect on the training of the model, dramatically reducing the number of epochs required. It can also have a regularizing effect, reducing generalization error much like the use of activation regularization.  

Batch normalization can be implemented during training by calculating the mean and standard deviation of each input variable to a layer per mini-batch and using these statistics to perform the standardization.  

In the case of deep recurrent neural networks, the inputs of the *n<sup>th</sup>* layer are the combination of *n-1<sup>th</sup>* layer, which is not raw feature inputs. As the training processes the effect of normalization or whitening reduces respectively, which causes the vanishing gradient problem. This can slow down the entire training process and cause saturation. To better training process, `batch normalization` is then applied to the internal layers of the deep neural networks. This approach ensures `faster convergence` in theory and during an experiment on benchmarks. In batch normalization, the features of a layer are independently normalized with mean zero and variance one. The algorithm of Batch normalization is given in Algorithm 1.


<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/algorithm1.png" style="width:80%">
</p>

The parameters &gamma; and &beta; are used for the `scale and shift factor` for the normalization values, so normalization does not only depend on layer values. If you use normalization techniques, the following criterions are recommended to consider during implementation:
- Increase the learning rate  
- Dropout (batch normalization does the same job)  
- *L<sub>2</sub>* weight regularization  
- Accelerating the learning rate decay  
- Remove Local Response Normalization (LRN) (if you used it)  
- Shuffle training sample more thoroughly  
- Useless distortion of images in the training set  

<br>


### 4.5. Activation Function  

The traditional Sigmoid and Tanh activation functions have been used for implementing neural network approaches in the past few decades. The graphical and mathematical representation is shown in Figure 1.

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure1.png" style="width:70%">
  <figcaption>Fig.1 - Activation function: (a) Sigmoid function, (b) hyperbolic transient.</figcaption>
</p>

#### (1) Sigmoid  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation2.png" style="width:20%">
</p>

#### (2) Tanh  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation3.png" style="width:20%">
</p>

#### (3) ReLU
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure2.png" style="width:30%">
  <figcaption>Fig.2 - ReLU (Rectified Linear Unit).</figcaption>
</p>

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation4.png" style="width:20%">
</p>

The popular activation function called ReLU (Rectified Linear Unit) proposed in 2010 `to solve the vanishing gradient problem` for training deep learning approaches. The basic concept is simple to keep all the values above zero and sets all negative values to zero that is shown in Figure2.  
(The ReLU activation was first used in AlexNet)  

As the activation function plays a crucial role in learning and weights for deep architectures, many researchers focus here because there is much that can be done in this area. There are several improved versions of ReLU that have been proposed, which provide even better accuracy compared to the ReLU activation function.  
- PReLU (Parametric ReLU), Leaky ReLU, ELU (Exponential Linear Unit), MELU (Multiple Exponent Linear Unit), S shape ReLU.  

#### (4) Leaky ReLU  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation5.png" style="width:20%">
</p>
(here *a* is a constant, the value is 0.1)

#### (5) ELU
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation6.png" style="width:30%">
</p>

<br>

### 4.6. Sub-Sampling Layer or Pooling Layer  

At present, two different techniques have been used for the implementation of deep networks in `the sub-sampling or pooling layer`: `Average and max-pooling`.
- Average pooling: Used for the first time in LeNet.  
- Max pooling: Used for the fist time in AlexNet.  
- Spatial pyramid pooling, multi-scale pyramid pooling, Fractional max pooling.  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure3.png" style="width:60%">
  <figcaption>Fig.3 - Average and max-pooling operations.</figcaption>
</p>

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure4.png" style="width:60%">
  <figcaption>Fig.4 - Spatial pyramid pooling.</figcaption>
</p>

<br>

### 4.7. Regularization Approaches for DL  

There are different regularization approaches that have been proposed in the past few years for deep CNN. The simplest but efficient approach called `dropout` was proposed by Hinton in 2012.  

**Dropout:** A randomly selected subset of activations is set to zero within a layer.  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure5.png" style="width:60%">
  <figcaption>Fig.5 - The concept of Dropout.</figcaption>
</p>

Another regularization approach is called Drop Connect.  

**Drop connect:** Instead of dropping the activation, `the subset of weights within the network layers are set to zero`. As a result, each layer receives the randomly selected subset of units from the immediate previous layer.

<br>

### 4.8. Optimization Methods for DL  

There are different optimization methods such as SGD, Adagrad, AdaDelta, RMSprop and Adam.

#### (1) Adagrad  
The main contribution was to calculate `adaptive learning rate during training`. For this method, the summation of the magnitude of the gradient is considered to calculate the adaptive learning rate. In the case with a large number of epochs, the summation of the magnitude of the gradient becomes large. The result of this is the learning rate decreases radically, which causes the gradient to approach zero quickly. The main drawback of this approach is that it causes problems during training.  

#### (2) RMSprop  
Proposed considering only the magnitude of the gradient of the immediately previous iteration, which prevents the problems with Adagrad and provides better performance in some cases.  

#### (3) Adam  
Proposed based on `the momentum` and `the magnitude of the gradient` for calculating and adaptive learning rate similar RMSprop. Adam has improved overall accuracy and helps for efficient training with the better convergence of deep learning algorithms.  

#### (4) EVE  
The improved version of the Adam optimization. Provides even better performance with fast and accurate convergence.  
