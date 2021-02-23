---
title:  "Deep Learining survey_4.CNN_acthitectures"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the fourth part (CNN section) of summary of a survey paper.
last_modified_at: 2021-02-23T08:06:00-05:00
---


This post is the fourth part (CNN section) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
This part is the basics of CNNs so, if you have already known them, you can skip this part.


### 3.2. Popular CNN Architectures

In this section, several popular state-of-the-art CNN architectures will be examined. In general, most deep convolutional neural networks are made of a key set of basic layers, including the convolution layer, the sub-sampling layer, dense layers, and the soft-max layer. The architectures typically consist of stacks of several convolutional layers and max-pooling layers followed by a fully connected and SoftMax layers at the end.  
 - LeNet, **AlexNet**, **VGG Net**, NiN  
 - **DenseNet**, **FractalNet**, **GoogLeNet**, Inception units, Residual Networks  
(Bold: the most popular architectures because of their state-of-the-art performance)
(* Fractal Net is an alternative of ResNet model)

- The baisc building components (convolution and pooling) are almost the same across these architectures. However, some topological differences are observed in the modern deep learning architectures.


#### 3.2.1. LeNet (1998)  

- Limited computation capability and memory capacity made the algorithm difficult to implement until about 2010.  
- LeCun proposed CNNs with the back-propagation algorithm and experimented on handwritten digit dataset to achieve state-of-the-art accuracy.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure3.png" style="width:40%">
  <figcaption>Fig.3 - The architecture of LeNet.</figcaption>
</p>



#### 3.2.2. AlexNet (2012)  

- In 2012, Alex Krizhevesky proposed a deeper and wider CNN model compared to LeNet and won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012.
- It was a significant breakthrough in the field of machine learning and computer vision for visual recognition and classification tasks and is the point in history where interest in deep learning increased rapidly.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure4.png" style="width:40%">
  <figcaption>Fig.4 - The architecture of AlexNet.</figcaption>
</p>

- Two new concepts, Local Response Normalization (LRN) and dropout, are introduced in this network.  
- LRN can be applied in two different ways:  
1) applying on single channel or feature maps  
2) applying across the channels or feature maps (neighborhood along the third dimension but a single pixel or location)  



#### 3.2.3. ZFNet / Clarifai (2013)  

- In 2013, Matthew Zeiler and Rob Fergue won the 2013 ILSVRC with a CNN architecture which was an extension of AlexNet.  
- As CNNs are expensive computationally, an optimum use of parameters is needed from a model complexity point of view.  
- ZFNet uses 7x7 kernels instead of 11x11 kernels to significantly reduce the number of weights.  
- This reduces the number of network parameters dramatically and improves overall recognition accuracy.  



#### 3.2.4. Network in Network (NiN)  

- This model is slightly different from the previous models where a couple of new concepts are introduced.  
1) multilayer perception convolution
  - convolutions are performed with 1x1 filter that help to add more nonlinearity in the models.  
  - This helps to increase the depth of the network, which can then be regularized with dropout.  
  - This concept is used often in the bottleneck layer of a deep learning model.  
2) Global Average Pooling (GAP) as an alternative of fully connected layers.  
  - This helps to reduce the number of network parameters significantly.  
  - By applying GAP on a large feature map, we can generate a final low dimensional feature vector without reducing the dimension of the feature maps.  



#### 3.2.5. VGGNET (2014)  

- The Visual Geometry Group (VGG), was the runner-up of the 2014 ILSVRC.  
- The main contribution of this work is that it shows that the depth of a network is a critical component to achieve better recognition or classification accuracy in CNNs.  
- Three VGG-E models, VGG-11, VGG-16 and VGG-19 were proposed the model had 11, 16 and 19 layers respectively.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure5.png" style="width:40%">
  <figcaption>Fig.5 - The basic building block of VGG network.</figcaption>
</p>

- All versions of the VGG-E models ended the same with three fully connected layers.  




#### 3.2.6. GoogLeNet (2014)  

- GoogLeNet, the winner of ILSVRC 2014, was a model proposed by Christian Szegedy of Google with the objective of reducing computation complexity compared to the traditional CNN.  
- The proposed method was to incorporate **Inception Layers** that had variable receptive fields, which were created by different kernel sizes.
- These receptive fields created operations that captured sparse correlation patterns in the new feature map stack.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure6.png" style="width:40%">
  <figcaption>Fig.6 - Inception layer: Naive version.</figcaption>
</p>

The initial concept of the Inception layer can be seen in Figure 6. GoogLeNet improved state-of-the-art recognition accuracy using a stack of Inception layers, seen in Figure 7.

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure7.png" style="width:40%">
  <figcaption>Fig.7 - Inception layer: Naive version.</figcaption>
</p>


- The difference between the naive inception layer and final Inception layer was the addition of 1x1 convolution kernels.  
- This kernels allowed for dimensionality reduction before computationally expensive layers.  
- GoogLeNet consisted of 22 layers in total, which was far greater than any network before it, but the number of network parameters was much lower than its predecessor AlexNet or VGG.  




#### 3.2.7. Residual Network (ResNet in 2015)  

- The winner of ILSVRC 2015 was the Residual Network architecture, ResNet.  
- Resnet was developed with the intent of designing ultra-deep networks that did not suffer from the vanishing gradient problem that predecessors had.  
- ResNet is developed with many different numbers of layers: 34, 50, 101, 152 and even 1202. The popular ResNet50 contained 49 convolution layers and 1 fully connected layer at the end of the network.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure8.png" style="width:40%">
  <figcaption>Fig.8 - Basic diagram of the Residual block.</figcaption>
</p>

- The basic block diagram of the ResNet architecture is shown in Figure8.  
- The output of a residual layer can be defined based on the outputs of (l-1)^th which comes from the previous layer defined as x_l-1.  
- F(x_l-1) is the output after performing various operations (e.g., convolution, Batch Normalization, activation function), the final output of residual unit is x_l which can be defined with the following equation:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/equation6.png" style="width:10%">
</p>

- The residual network consists of several basic residual blocks, but the operations in the residual block can be varied depending on the different architecture of residual networks.  
- Recently, some other variants of residual models have been introduced based on the Residual Network architecture.  (there are several advanced architectures that are combined with inception and residual units which can be seen in Figure9)

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure9.png" style="width:40%">
  <figcaption>Fig.9 - The basic block diagram for Inception Residual unit.</figcaption>
</p>

- The concept of Inception block with residual connections is introduced in the Inception-v4 architecture.  



#### 3.2.8. Densely Connected Network (DenseNet)  

- DenseNet consists of densely connected CNN layers, the outputs of each layer are connected with all successor layers in a dense block.  
- This concept is efficient for feature resue, which dramatically reduces network parameters.  **(WHY???)**  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure10.png" style="width:40%">
  <figcaption>Fig.10 - A 4-layer Dense block with a growth rate of k=3.</figcaption>
</p>

- The l^th layer received all the feature maps from previous layers as input:

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/equation7.png" style="width:10%">
</p>

- DenseNet consists of several dense blocks and transition blocks, which are placed between two adjacent dense blocks.  
- H_l(.) performs three different consecutive operations: Batch-Normalization, followed by a ReLU and a 3x3 convolution operation.
- In the transation block, 1x1 convolutional operations are performed with BN followed by a 2x2 average pooling layer.  
- This new model shows state-of-the-art accuracy with a reasonable number of network parameters for object recognitions tasks.  



#### 3.2.9. FractalNet (2016)  

- This architecture is an advanced and alternative architecture of ResNet model, which is efficient for designing large models with nominal depth, but shorter paths for the propagation of gradient during training.  
- This concept is based on drop-path which is another regularization approach for making large networks.  
- As a result, this concept helps to enforce speed versus accuracy tradeoffs.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure11.png" style="width:40%">
  <figcaption>Fig.11 - The detailed FractalNet module on the left and FractalNet on the right.</figcaption>
</p>



#### 3.3. CapsuleNet  
