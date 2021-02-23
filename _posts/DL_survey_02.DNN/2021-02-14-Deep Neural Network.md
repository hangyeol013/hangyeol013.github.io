---
title:  "Deep Learining survey_2.Deep Neural networks"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the second part (Deep Neural Network) of summary of a survey paper.
last_modified_at: 2021-02-14T08:06:00-05:00
---


This post is the second part (Deep Neural Network) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
The main objective of this work is to provide an overall idea on deep learning and its related fields.



### 2.1 The history of DNN

A brief history of neural networks highlighting key event is, as shown in Figure 8

<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/Figure8.png" style="width:80%">
  <figcaption>Fig.1 - The history of deep learning development.</figcaption>
</p>

**Artificial neurons:** The fundamental component of building ANNS which try to mimic the behavior of the human brain

**Perceptron:** A basic computational element which receives inputs from external sources and has some internal parameters which produce outputs.
 - It's also called a node or unit.
 - ANNs or general NNs consist of Multilayer Perceptrons (MLP) which contain one or more hidden layers with multiple hidden units in them.


Here, I wrote the meaning of key words briefly, I'll add the details later.

### 2.2 Gradient Descent
- A first-order optimization algorithm which is used for finding the local minima of an objective function.
- This has been used for training ANNs in the last couple of decades successfully.

### 2.3 Stochastic Gradient Descent (SGD)  
- Since a long training time is the main drawback for the traditional gradient descent approach, the SGD approach is used for training DNNs.

### 2.4 Back-propagation (BP)  
- In the case of MLPs, we can easily represent NN models using computation graphs which are directive acyclic graphs.
- For that representation of DL, we can use the chain-rule to efficiently calculate the gradient from the top to the bottom layers with BP.

### 2.5 Momentum  
- A method which helps to accelerate the training process with the SGD approach.
- main idea: using the moving average of the gradient instead of using only the current real value of the gradient.
- we can express this with the following equation mathematically:


<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/equation1.png" style="width:30%">
</p>

- here r is the momentum and n is the learning rate for the tth round of training.
- The main advantage of using momentum during training is to prevent the network from getting stuck in local minimum.
 (How??? explain with equation)
- a higher momentum value overshoots its minimum, possibly making the network unstable.
- In general r is set to 0.5 until the initial learning stabilizes and is then increased to 0.9 or higher.

### 2.6 Learning rate
- the step size considered during training which makes the training process faster.
- selecting the value of the learning rate is sensitive.  
 ex)  
     a higher value -> the network may start diverging instead of converging.  
     a smaller value -> it will take more time for the network to converge. (+ it may easily get stuck in minima)  
 -> Typical solution: to reduce the learning rate during training.  

 Three common approaches for reducing the learning rate:  
1) Constant decay: we can define a constant which is applied to reduce the learning rate manually with a defined step function.  
2) Factored decay: the learning rate can be adjusted during training with the following equation:  
<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/equation2.png" style="width:10%">
</p>
n_t: the tth round learning rate,  
n_0: the initial learning rate,  
$\beta$: the decay factor with a value between the range of (0,1)  
3) Exponential decay
<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/equation3.png" style="width:10%">
</p>
- The common practice is to use a learning rate decay of beta=0.1 to reduce the learning rate by a factor of 10 at each stage.

### 2.7 Weight decay
- used for training deep learning models as an L2 regularization approach, which helps to prevent overfitting the network and model generalization.
- L2 regularization for F(theta, x) can be defined as,
 <p>
   <img src="/assets/images/blog/DL_survey_02.DNN/equation4.png" style="width:30%">
 </p>
- the gradient for the weight theta is:
 <p>
   <img src="/assets/images/blog/DL_survey_02.DNN/equation5.png" style="width:15%">
 </p>
- General practice is to use the value $/lambda$ = 0.0004.
- A smaller lambda will accelerate training.

**Other necessary components for efficient training**
- data preprocessing
- augmentation
- network initialization approaches
- batch normalization
- activation functions
- regularization with dropout
- different optimization approaches


Loss function
Optimization - Back propagation
Learning rate - over fitting under fitting
Regularization
