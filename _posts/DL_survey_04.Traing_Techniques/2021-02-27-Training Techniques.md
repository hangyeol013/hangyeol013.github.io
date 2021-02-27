---
title:  "Deep Learining survey_5.Advanced Training Techniques"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the fifth part (Advanced Training Techniques) of summary of a survey paper.
last_modified_at: 2021-02-27T08:06:00-05:00
---


This post is the fifth part (Advanced Training Techniques) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
The main objective of this work is to provide an overall idea on deep learning and its related fields.


The advanced training techniques or components which need to be considered carefully for efficient training of DL approach. The techniques include input pre-processing, a better initialization method, batch normalization, alternative convolutional approaches, advanced activation functions, alternative pooling techniques, network regularization approaches, and better optimization method for training.  
This following sections are discussed on individual advanced training techniques individually.  


### 4.1. Preparing Dataset  

Presently different approaches have been applied before feeding the data to the network. The different operations to prepare a dataset are as follows; sample rescaling, mean subtraction, random cropping, flipping data with respect to the horizon or vertical axis, color jittering, PCA/ZCA whitening and many more.  

<br>

### 4.2. Network Initialization  

- The initialization of deep networks has a big impact on the overall recognition accuracy.  
- For complex tasks with high dimensionality data training, a DNN becomes difficult because weights should not be symmetrical due to the back-propagation process.  
- Therefore, effective initialization techniques are important for training this type of DNN.  
- Xavier initialization approach  
- LSUV (Layer-sequential unit-invariance)  
- He initialization: The distribution of the weights of *l<sup>th</sup>* layer will be a normal distribution with mean zero and variance **2/n<sub>l</sub>** which can be expressed as follows:  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation1.png" style="width:20%">
</p>

<br>

### 4.3. Batch Normalization  

- Batch normalization helps accelerate DL processes by reducing internal covariance by shifting input samples.  
(what that means is the inputs are linearly transformed to have zero mean and unit variance)  
- For whitened inputs, the network converges faster and shows better regularization during training, which has an impact on the overall accuracy.  
- In the case of deep recurrent neural networks, the inputs of the *n<sup>th</sup>* layer are the combination of *n-1<sup>th</sup>* layer, which is not raw feature inputs.  
- As the training processes the effect of normalization or whitening reduces respectively, which causes the vanishing gradient problem.  
- This can slow down the entire training process and cause saturation.  
- To better training process, batch normalization is then applied to the internal layers of the deep neural networks.  
- This approach ensures faster convergence in theory and during an experiment on benchmarks.  
- In batch normalization, the features of a layer are independently normalized with mean zero and variance one.  
- The algorithm of Batch normalization is given in Algorithm 1.


<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/algorithm1.png" style="width:60%">
</p>

The parameters $\gamma$ and $\beta$ are used for the scale and shift factor for the normalization values, so normalization does not only depend on layer values. If you use normalization techniques, the following criterions are recommended to consider during implementation:
- Increase the learning rate  
- Dropout (batch normalization does the same job)  
- *L<sub>2</sub>* weight regularization  
- Accelerating the learning rate decay  
- Remove Local Response Normalization (LRN) (if you used it)  
- Shuffle training sample more thoroughly  
- Useless distortion of images in the training set  


### 4.4. Alternative Convolutional Methods  

- Alternative and computationally efficient convolutional techniques that reduce the cost of multiplications by a factor of 2.5 have been proposed.  


### 4.5. Activation Function  

- The traditional Sigmoid and Tanh activation functions have been used for implementing neural network approaches in the past few decades.  
- The graphical and mathematical representation is shown in Figure 1.

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure1.png" style="width:50%">
  <figcaption>Fig.1 - Activation function: (a) Sigmoid function, (b) hyperbolic transient.</figcaption>
</p>

**Sigmoid:**  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation2.png" style="width:20%">
</p>

**Tanh:**  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation3.png" style="width:20%">
</p>


The popular activation function called ReLU (Rectified Linear Unit) proposed in 2010 to solve the vanishing gradient problem for training deep learning approaches. The basic concept is simple to keep all the values above zero and sets all negative values to zero that is shown in Figure2.  
(The ReLU activation was first used in AlexNet)  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure2.png" style="width:30%">
  <figcaption>Fig.2 - ReLU (Rectified Linear Unit).</figcaption>
</p>

**ReLU:**  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation4.png" style="width:20%">
</p>

As the activation function plays a crucial role in learning and weights for deep architectures, many researchers focus here because there is much that can be done in this area. There are several improved versions of ReLU that have been proposed, which provide even better accuracy compared to the ReLU activation function.  
- PReLU (Parametric ReLU), Leaky ReLU, ELU (Exponential Linear Unit), MELU (Multiple Exponent Linear Unit), S shape ReLU.  

**Leaky ReLU:**  
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation5.png" style="width:20%">
</p>
(here *a* is a constant, the value is 0.1)

**ELU:**
<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/equation6.png" style="width:20%">
</p>



### 4.6. Sub-Sampling Layer or Pooling Layer  

At present, two different techniques have been used for the implementation of deep networks in the sub-sampling or pooling layer: Average and max-pooling.
- Average pooling: Used for the first time in LeNet.  
- Max pooling: Used for the fist time in AlexNet.  
- Spatial pyramid pooling, multi-scale pyramid pooling, Fractional max pooling.  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure3.png" style="width:50%">
  <figcaption>Fig.3 - Average and max-pooling operations.</figcaption>
</p>

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure4.png" style="width:50%">
  <figcaption>Fig.4 - Spatial pyramid pooling.</figcaption>
</p>


### 4.7. Regularization Approaches for DL  

There are different regularization approaches that have been proposed in the past few years for deep CNN. The simplest but efficient approach called dropout was proposed by Hinton in 2012.  
**Dropout:** A randomly selected subset of activations is set to zero within a layer.  

<p>
  <img src="/assets/images/blog/DL_survey_04.Training_Techniques/Figure5.png" style="width:50%">
  <figcaption>Fig.5 - The concept of Dropout.</figcaption>
</p>

Another regularization approach is called Drop Connect.  
**Drop connect:** Instead of dropping the activation, the subset of weights within the network layers are set to zero. As a result, each layer receives the randomly selected subset of units from the immediate previous layer.



### 4.8. Optimization Methods for DL  

There are different optimization methods such as SGD, Adagrad, AdaDelta, RMSprop and Adam.

**Adagrad)**  
- The main contribution was to calculate adaptive learning rate during training. For this method, the summation of the magnitude of the gradient is considered to calculate the adaptive learning rate.  
- In the case with a large number of epoches, the summation of the magnitude of the gradient becomes large.  
- The result of this is the learning rate decreases radically, which causes the gradient to approach zero quickly.  
- The main drawback of this approach is that it causes problems during training.  
<br>
**RMSprop)**  
- Proposed considering only the magnitude of the gradient of the immediately previous iteration, which prevents the problems with Adagrad and provides better performance in some cases.  
<br>
**Adam)**  
- Proposed based on the momentum and the magnitude of the gradient for calculating and adaptive learning rate similar RMSprop.  
- Adam has improved overall accuracy and helps for efficient training with the better convergence of deep learning algorithms.  
<br>
**EVE)**
- The improved version of the Adam optimization.  
- Provides even better performance with fast and accurate convergence.  
