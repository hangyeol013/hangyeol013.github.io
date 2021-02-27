---
title:  "Deep Learining survey_1.Introduction"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the first part (introduction section) of summary of a survey paper.
last_modified_at: 2021-02-14T08:06:00-05:00
---


This post is the first part (introduction section) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
The main objective of this work is to provide an overall idea on deep learning and its related fields.


I'll organize deep learning concepts according to the history of deep learning with keywords.  
Before we get the concepts of Deep learning, we will look over the basic concepts from learning.


**Learning:** A procedure consisting of estimating the model parameters so that the learned model (algorithm) can perform a specific task.


### 1.1 Deep learning approaches can be categorized as follows:
Supervised, Semi-supervised (partially supervised), unsupervised  
(Another category of learning approach called Reinforcement Learning, Deep RL)

**Supervised learning:** A learning technique that uses labeled data.   
 - the environment has a set of inputs and corresponding outputs
 - DNN, CNN, RNN (LSTM, GRU)

**Semi-supervised learning:** learning that occurs based on partially labeled datasets.  
 - In some cases, DRL and Generative Adversarial Networks are used as semi-supervised learning technique
 - RNN (LSTM, GRU)

**Unsupervised learning systems:** ones that can without the presence of data labels.  
 - The agent learns the internal representation or important features to discover unknown relationships or structure within the input data.
 - Clustering, Dimensionality reduction, Generative techniques
 - Auto-Encoders (AE), Resticted Boltzmann Machines (RBM), GAN, RNNs (LSTM, GRU)

**Deep reinforcement learning:** A learning technique for use in unknown environments.  
 - Do not have a straight forward loss function, thus making learning harder compared to traditional supervised approaches

 *** The fundamental differences between RL and supervised learning:**  
 (1) Do not have full access to the function you are trying to optimize (you must query them through interaction)  
 (2) interacting with a state-based environment (Input depends on previous actions)  


<p>
  <img src="/assets/images/blog/DL_survey_01.introduction/Figure2.png" style="width:40%">
  <figcaption>Fig.1 - Category of deep learning approaches.</figcaption>
</p>


### 1.2 Feature Learning

**Definition)**  
**Machine Learning:** An application of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed  
**Deep Learning:** A subfield of machine learning based on artificial neural networks with representation learning.


A key difference between traditional ML and DL is in how features are extracted.  
**Traditional ML:** handcrafted engineering features by applying several feature extraction algorithm, and then apply the learning algorithms  
**DL:** the features are learned automatically and are represented hierarchically in multiple levels.


Table 1 shows the difference feature-based learning approaches with different learning steps.
<br>

<p>
  <img src="/assets/images/blog/DL_survey_01.introduction/Table1.png" style="width:80%">
  <figcaption>Table.1 - Different feature learning approaches.</figcaption>
</p>


### 1.3 Why DL?

1) Universal Learning Approach: It can be applied to almost any application domain  
2) Robust: Do not require the precisely designed feature.  
3) Generalization: The same DL approach can be used in different applications or with different data types.  
- This approach is often called transfer learning
- This approach is helpful where the problem does not have sufficient available data.

4) Scalability: The DL approach is highly scalable  
- Ex) ResNet contains 1202 layers and is often implemented at a supercomputing scale.  


I will skip 1.4, 1.5 contents, you can see them on page 4-7.  


### 1.6 Challenges of DL

In this section, I just wrote the keywords, and in this paper, it says that these below challenges have already been considered by the DL community.  
Moreover, you can see several survey papers based on several learning approaches from this section.

(1) Big data analytics using DL  
(2) Scalability of DL approaches  
(3) Ability to generate data  
(4) Energy efficient techniques for special purpose devices  
(5) Multi-task and transfer learning or multi-module learning  
(6) Dealing with causality in learning  