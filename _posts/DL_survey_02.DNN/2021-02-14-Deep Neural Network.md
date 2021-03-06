---
title:  "Deep Learning survey_2.Deep Neural networks"
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

A brief history of neural networks highlighting key event is, as shown in Figure 1

<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/Figure1.png" style="width:80%">
  <figcaption>Fig.1 - The history of deep learning development.</figcaption>
</p>

**Artificial neurons:** `The fundamental component of building ANNs` which try to mimic the behavior of the human brain.

**Perceptron:** `A basic computational element` which receives inputs from external sources and has some `internal parameters` which produce outputs.
 - It's also called a `node` or `unit`.
 - ANNs or general NNs consist of `Multilayer Perceptrons` (MLP) which contain one or more hidden layers with multiple hidden units in them.


Here, I wrote the meaning of key words briefly, I'll post on other page about optimizations and regularizations later.  

### 2.2 Gradient Descent
- A first-order optimization algorithm which is used for finding the local minima of an objective function.
**optimization:** The task of either minimizing or maximizing some function *f(x)* by altering x.  
**objective function:** The function we want to minimize or maximize  
(=Cost function = Error function = Criterion)
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
*n<sub>t</sub>*: the tth round learning rate,  
*n<sub>0</sub>*: the initial learning rate,  
$ \beta $: the decay factor with a value between the range of (0,1)  
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



### +) Deep learning written by Ian Goodfellow.  

This contents are from deep learning book (chapter5. Machine learning basics) written by Ian Goodfellow.  

#### 1.Learning algorithms  
If a computer program's performance at tasks in T, as measured by P, improves with experience E, it is said to learn form experience E with respect to some class of tasks T and performance measure P.  

**The task, T)**  
- Classification  
- Regression  
- Transcription  
- Machine Translation  
- Anomaly detection  
- Synthesis and sampling  
- Denoising  
- Density / Probability mass function estimation  


**The performance Measure, P)**  

(1) Accuracy: The proportion of examples for which the model produces the correct output.  
(2) Error rate: the proportion of examples for which the model produces an incorrect output.  
 - Regression: Mean Squared Error  
 - Binary classification: Sigmoid -> Binary cross entropy  
 - Multi classification: Softmax -> Cross entropy  

**The Experience, E)**  
- Supervised Learning  
- Unsupervised Learning  
- Reinforcement Learning  

#### 2.Overfitting and Underfitting  

**Generalization:** The ability to perform well on previously unobserved inputs.  
**Underfitting:** Occurs when the model is not able to obtain a sufficiently low error value on the training set.  
**Overfitting:** Occurs when the gap between the training error and the test error is too large.  
**Regularization:** Any modification we make to a learning algorithm that is intended to reduce its generalization error (=test error) but not its training error (ex. weight decay)  

<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/Figure2.png" style="width:60%">
  <figcaption>Fig.2 - Overfitting, Underfitting.</figcaption>
</p>

You can see the relationship between capacity and error in figure 3. Training and test error behave differently. At the left end of the graph, training error and generalization error are both high. This is the `underfitting regime`. As we increase capacity, training error decreases, but the gap between training and generalization error increases. Eventually, the size of this gap outweighs the decrease in training error, and we enter the `overfitting regime`, where capacity is too large, above the optimal capacity.  

<p>
  <img src="/assets/images/blog/DL_survey_02.DNN/Figure3.png" style="width:60%">
  <figcaption>Fig.3 - Typical relationship between capacity and error.</figcaption>
</p>


#### 3. Hyperparameters and Validation sets  

**Hyperparameters:** Settings that we can use to control the algorithm's behavior.  
(=Learning rate, Loss function, Epoch, ...)  
(The values of hyperparameters are not adapted by the learning algorithm itself.)  

If hyperparameters are learned on the training set, such hyperparameters would always choose the maximum possible model capacity, resulting in overfitting.  

**validation set:** Part of train data (train/test) which is used to estimate the generalization error during or after training, allowing for the hyperparameters to be updated accordingly. (Used to estimate the generalization error during training)  


#### 4. Stochastic Gradient Descent  

A recurring problem in machine learning is that large training sets are necessary for good generalization, but large training sets are also more computationally expensive.  
The cost function used by a machine learning algorithm often decomposes as a sum over training examples of some per-example loss function. As the training set size grows to billions of examples, the time to take a single gradient step becomes prohibitively long (in the case of gradient descent).  

**Stochastic Gradient descent:** On each step of the algorithm, we can sample a minibatch of examples drawn uniformly from the training set. The minibatch size is typically chosen to be a relatively small number of examples, ranging from 1 to a few hundred. The stochastic gradient descent algorithm follows the gradient descent algorithm with this small minibatch.  
