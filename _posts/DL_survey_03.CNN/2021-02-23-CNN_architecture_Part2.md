---
title:  "Deep Learning survey_5.CNN_acthitectures_Part.2(GoogLeNet, ResNet, DenseNet)"
search: true
categories:
  - Deep learning
summary: This post is the fifth part (popular CNN architecture section) of summary of a survey paper.
toc: true
toc_sticky: true
header:
  teaser: /assets/images/thumbnails/thumb_basic.jpg
last_modified_at: 2021-02-23T08:06:00-05:00
---


This post is the fifth part (popular CNN architecture section) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
This part is the basics of CNNs so, if you have already known them, you can skip this part.


### 3.2. Popular CNN Architectures

In this section, several popular state-of-the-art CNN architectures will be examined. In general, most deep convolutional neural networks are made of a key set of basic layers, including the `convolution layer`, the `sub-sampling layer`, `dense layers`, and the `soft-max layer`. The architectures typically consist of stacks of several convolutional layers and max-pooling layers followed by a fully connected and SoftMax layers at the end.  
- LeNet, **AlexNet**, **VGG Net**, NiN  
- **DenseNet**, **FractalNet**, **GoogLeNet**, Inception units, Residual Networks  
(Bold: the most popular architectures because of their state-of-the-art performance)  
(* Fractal Net is an alternative of ResNet model)

The basic building components (convolution and pooling) are almost the same across these architectures. However, some topological differences are observed in the modern deep learning architectures. In the last article, we looked over LeNet, AlexNet and VGGNet. In this article, we will look over following several models concentrating upon GoogleNet, ResNet.


#### 3.2.6. GoogLeNet (2014)  

- GoogLeNet, the winner of ILSVRC 2014, was a model proposed by Christian Szegedy of Google with the objective of reducing computation complexity compared to the traditional CNN.  
- It is also called `Inception v1` as there are v2, v3 and v4 later on.  
- The proposed method was to incorporate *Inception Layers* that had variable receptive fields, which were created by different kernel sizes.
- These receptive fields created operations that captured sparse correlation patterns in the new feature map stack.  

We will cover following below orders:  
(These are from [Sik-Ho Tsang@Medium](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7))  
1. The 1x1 Convolution  
2. Inception Module  
3. Global Average Pooling  
4. Overall Architecture  


###### 1. The 1x1 Convolution  

The 1x1 Convolution is introduced by NIN (Network In Network). Originally, NIN uses it for introducing more non-linearity to increase the representational power of the network since authors in NIN believe data is in non-linearity form. In GooLeNet, 1x1 Convolution is used as a dimension reduction module to reduce the computation. By reducing the computation bottleneck, depth and width can be increase.  


* 5x5 convolution without the use of 1x1 convolution:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure22.png" style="width:60%">
  <figcaption>Fig.2 - Without the use of 1x1 convolution.</figcaption>
</p>

Number of operations = (14x14x48) x (5x5x480) = 112.9M


* 5x5 convolution with the use of 1x1 convolution:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure23.png" style="width:60%">
  <figcaption>Fig.2 - With the use of 1x1 convolution.</figcaption>
</p>

Number of operations for 1x1 = (14x14x16) x (1x1x480) = 1.5M  
Number of operations for 5x5 = (14x14x48) x (5x5x16) = 3.8M  
Total Number of operations = 1.5M + 3.8< = 5.3M  
Which is much smaller than 112.9M.  

Thus, inception module can be built without increasing the number of operations largely compared the one without 1x1 convolution. 1x1 convolution can help to reduce model size which can also somehow help to reduce the overfitting problem.  

<br>

###### 2. Inception Module  

The inception module (naive version, without 1x1 convolution) is as below:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure6.png" style="width:60%">
  <figcaption>Fig.3 - Inception layer: Naive version.</figcaption>
</p>

1x1 conv, 3x3 conv, 5x5 conv and 3x3 max pooling are done altogether for the previous input, and stack together again at output. When image's coming in, different sizes of convolutions as well as max pooling are tried. Then different kinds of features are extracted. After that, all feature maps at different paths are concatenated together as the input of the next module. However, without the 1x1 convolution, we can imagine how large the number of operation is.  


<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure7.png" style="width:60%">
  <figcaption>Fig.4 - Inception layer: With dimensionality reduction.</figcaption>
</p>

Thus, 1x1 convolution is inserted into the inception module for dimension reduction.  


###### 3. Global Average Pooling  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure24.png" style="width:60%">
  <figcaption>Fig.4 - Fully Connected Layer VS Global Average Pooling.</figcaption>
</p>

Previously, fully connected (FC) layers are used at the end of network, all inputs are connected to each output. In GoogLeNet, global average pooling is used nearly at the end of network by averaging each feature map from 7x7 to 1x1.  

Number of weights (FCL) = 7x7x1024x1024 = 51.3M  
Number of weights (GAP) = 0 (Average Pooling doesn't need weights)  

And authors found that a move from FC layers to average pooling improved the top-1 accuracy by about 0.6%.  


###### 4. Overall Architecture  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure25.png" style="width:60%">
  <figcaption>Fig.5 - GoogLeNet overall architecture (From Left to Right).</figcaption>
</p>

There are 22 layers in total. We can see that there are numerous inception modules connected together to go deeper. There are some intermediate softmax branches at the middle. This intermediate softmax branches are used for training only. These branches are auxiliary classifiers which consist of:  
5x5 average pooling (3 stride)  
1x1 conv (128 filters)  
1024 FC  
1000 FC  
softmax  

The loss is added to the total loss, with weight 0.3. Authors claim it can be used for combating gradient vanishing problem, also providing regularization. It is not used in testing or inference time.  

Below is the details about the parameters of each layer.

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Table2.png" style="width:60%">
  <figcaption>Table.1 - Details about parameters of each layer in GoogLeNet Network.</figcaption>
</p>

<br>


#### 3.2.7. Residual Network (ResNet in 2015)  

- The winner of ILSVRC 2015 was the Residual Network architecture, ResNet.  
- Resnet was developed with the intent of designing ultra-deep networks that did not suffer from the vanishing gradient problem that predecessors had.  
- ResNet is developed with many different numbers of layers: 34, 50, 101, 152 and even 1202. The popular ResNet50 contained 49 convolution layers and 1 fully connected layer at the end of the network.  

We will cover following below orders:  
(These are from [Sik-Ho Tsang@Medium](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8))  
1. Problem of Plain Network (Vanishing / Exploding gradient)  
2. Skip / Shortcut Connection in Residual Network (ResNet)  
3. ResNet Architecture  
4. Bottleneck Design  


###### 1. Problems of Plain Network  

For conventional deep learning networks, they usually have conv layers then fully connected layers for classification task, without any skip /shortcut connection, we call them plain networks here. When the plain network is deeper (layers are increased), the problem of vanishing / exploding gradients occurs.  

**Vanishing / Exploding Gradients**  

During backpropagation, when partial derivative of the error function with respect to the current weight in each iteration of training, this has the effect of multiplying n of these small / large numbers to compute gradients of the front layers in an n-layer network.  

When the network is deep, and multiplying n of these small numbers will become zero (vanished).  
When the network is deep, and multiplying n of these large numbers will become too large (exploded).  

We expect deeper network will have more accurate prediction. However, below shows an example, 20-layer plain network got lower training error and test error than 56-layer plain network, a degradation problem occurs due to vanishing gradient.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure26.png" style="width:60%">
  <figcaption>Fig.6 - Plain Networks for CIFAR-10 Dataset.</figcaption>
</p>



###### 2. Skip / Shortcut connection in residual network (ResNet)  

To solve the problem of vanishing/exploding gradient, a skip / shortcut connection is added to add the input x to the output after few weight layers as below:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure27.png" style="width:60%">
  <figcaption>Fig.7 - A Building Block of Residual Network.</figcaption>
</p>

Hence, the output *H(x)=F(x)+x*. The weight layers actually is to learn a kind of residual mapping: *F(x)=H(X)-x*.  

Even if there is vanishing gradient for the weight layers, we always still have the identity x to transfer back to earlier layers.  



###### 3. ResNet Architecture  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure28.png" style="width:60%">
  <figcaption>Fig.8 - 34-layer ResNet with Skip/Shortcut (top), 34-layer Plain Network (middle) and 19-layer VGG-19 (bottom).</figcaption>
</p>

For ResNet, there are 3 types of skip / shortcut connections when the input dimensions are smaller than the output dimensions.  

(a) Shortcut performs identity mapping, with extra zero padding for increasing dimensions. Thus, no extra parameters.  
(b) The projection shortcut is used for increasing dimensions only, the other shortcuts are identity. Extra parameters are needed.  
(c) All shortcuts are projections. Extra parameters are more than that of (b).  


###### 4. Bottleneck Design  

Since the network is very deep now, the time complexity is high. A bottleneck design is used to reduce the complexity as follows:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure29.png" style="width:60%">
  <figcaption>Fig.9 - The Basic Block (left) and the proposed bottleneck design (right).</figcaption>
</p>

The 1x1 conv layers are added to the start and end of network like being used in NIN and GoogLeNet. It turns out that 1x1 conv can reduce the number of connections (parameters) while not degrading the performance of the network so much.  

With the bottleneck design, 34-layer ResNet become 50-layer ResNet. And there are deeper network with the bottleneck design: ResNet-101 and ResNet-152. The overall architecture for all network is as below:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure30.png" style="width:60%">
  <figcaption>Fig.10 - The overall architecture for all network.</figcaption>
</p>

It is noted that VGG-16/19 has 15.3/19/6 billion FLOPS. ResNet-152 still has lower complexity than VGG-16/19.  



#### 3.2.8. Densely Connected Network (DenseNet)  

- DenseNet consists of densely connected CNN layers, the outputs of each layer are connected with all successor layers in a dense block.  
- This concept is efficient for feature reuse, which dramatically reduces network parameters.  **(WHY???)**  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure10.png" style="width:60%">
  <figcaption>Fig.10 - A 4-layer Dense block with a growth rate of k=3.</figcaption>
</p>

- The l<sup>th</sup> layer received all the feature maps from previous layers as input:

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/equation7.png" style="width:30%">
</p>

- DenseNet consists of several dense blocks and transition blocks, which are placed between two adjacent dense blocks.  
- H<sub>l</sub>(.) performs three different consecutive operations: Batch-Normalization, followed by a ReLU and a 3x3 convolution operation.
- In the transaction block, 1x1 convolutional operations are performed with BN followed by a 2x2 average pooling layer.  
- This new model shows state-of-the-art accuracy with a reasonable number of network parameters for object recognitions tasks.  



#### 3.2.9. FractalNet (2016)  

- This architecture is an advanced and alternative architecture of ResNet model, which is efficient for designing large models with nominal depth, but shorter paths for the propagation of gradient during training.  
- This concept is based on drop-path which is another regularization approach for making large networks.  
- As a result, this concept helps to enforce speed versus accuracy tradeoffs.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure11.png" style="width:50%">
  <figcaption>Fig.11 - The detailed FractalNet module on the left and FractalNet on the right.</figcaption>
</p>



#### 3.3. CapsuleNet  

- CNNs are an effective methodology for detecting features of an object and achieving good recognition performance compared to state-of-the-art handcrafted feature detectors.  
- There are **limits to CNNs**, which are that it does not take into account **special relationships**, **perspective**, **size**, and **orientation, of features**.  
- Imagine a neuron which contains the likelihood with properties of features (perspective, orientation, size etc.).  
- This special type of neurons, capsules, can detect face efficiently with distinct information.  
- The capsule network consists of several layers of capsule nodes.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure12.png" style="width:80%">
  <figcaption>Fig.12 - A CapsNet encoding unit with 3 layers.</figcaption>
</p>

- The entire encoding and decoding processes of CapsNet is shown in figures 12 and 13.  
- The primary capsuels are used 8x32 kernels which generates 32x8x6x6 (32 groups for 8 neurons with 6x6 size)  
- Even if a feature moves if it is still under a max pooling window it can be detected.  
- As the capsule contains the weighted sum of features from the previous layer, therefore this approach is capable of detecting overlapped features which is important for segmentation and detection tasks.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure12.png" style="width:80%">
  <figcaption>Fig.13 - The decoding unit where a digit is reconstructed from DigitCaps layer representation. The Euclidean distance is used minimizing the error between the input sample and the reconstructed sample from the sigmoid layer. True labels are used for reconstruction target during training.
  </figcaption>
</p>

- In the traditional CNN, a single cost function is used to evaluate the overall error which propagates backward during training.  
- However, in this case, if the weight between two neurons is zero, then the activation of a neuron is not propagated from that neuron.  
- The signal is routed with respect to the feature parameters rather than a one size fits all cost function in iterative dynamic routing with the agreement.  
- This new CNN architecture provides state-of-the-art accuracy for handwritten digit recognition on MNIST.  
- However, from an application point of view, this architecture is more suitable for segmentation and detection tasks compare to classification task.  
- CapsNet paper: [Dynamic routing between capsules](https://arxiv.org/pdf/1710.09829.pdf).


#### 3.4. Comparison of Different Models  

- The comparison of recently proposed models based on error, network parameters, and a maximum number of connections are given in Table 1.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Table1.png" style="width:80%">
  <figcaption>Table 1 - The top-5% errors with computational parameters and macs for different deep CNN models.  </figcaption>
</p>


#### 3.5. Other DNN Models  

- Xception (popular in the computer vision community), RCNN (Recurrent Convolution Neural Network), IRCNN (Inception Convolutional Recurrent Neural Networks, provided better accuracy compared RCNN and inception network with almost identical network parameters)
- ViP CNN (Visual Phase Guided CNN): Proposed with phase guided message passing a structure (PMPS) to build connections between relational components, which show better speed up and recognition accuracy.  
- FCN (Fully convolutional network): Proposed for segmentation tasks.  
- Pixel Net, A deep network with stochastic depth, deeply-supervised networks, and ladder network.  



#### 3.6. Applications of CNNs  


##### CNNs for Solving A Graph Problem)  
- Learning graph data structures is a common problem with various applications in data mining and machine learning tasks.  
- DL techniques have made a bridge in between the machine learning and data mining groups.  

##### Image Processing and Computer Vision)  

- There is a good survey on DL approaches for image processing and computer vision related tasks, including image classification, segmentation and detection: [Available online](https://github.com/kjw0612/awesome-deep-vision)  
- The DL approaches are massively applied to human activity recognition tasks and achieved state-of-the-art performance compared to exiting approaches.  
- However, the state-of-the-art models for classification, segmentation and detection task are listed as follows:  

**(1) Models for classification problems:**  
- The models with classification layer can be used as feature extraction for segmentation and detection tasks.  
- AlexNet, VGGNet, GoogleNet, ResNet, DenseNet, FractalNet, CapsuleNet, IRCNN, IRRCNN, DCRN, and so on...  

**(2) Models for segmentation problems:**  
- The segmentaion model consists of two units: Encoding and Decoding units.  
- The encoding unit: Convolution, Subsampling operations to encode to the lower dimensional latent space.    
- The decoding unit: Deconvolution, Up-sampling operation to decode the image from latent space.  
- FCN, SegNet, RefineNet, PSPNet, DeepLab, UNet and R2U-Net.  

**(3) Models for detection problems:**  
- The detection problem is a bit different compared to classification and segmentation problems.  
- In this case, the model goal is to identify target types with its corresponding position.  
- The model answers two questions: what is the object (classification problem)? and where is the object (regression problem)?
- To achieve these goals, two losses are calculated for classification and regression unit in top of the feature extraction module and the model weights are updated with respect to the both loses.  
- RCNN, fast RCNN, mask R-CNN, YOLO, SSD (single Shot MultiBox Detection) and UD-Net  


##### Speech Processing  

- CNNs are also applied to speech processing, such as speech enhancement using multimodal deep CNN, and audio tagging using CGRN (Convolutional Gated Reccurent Network).  


##### CNN for Medical Imaging  

- Several popular DL methods were developed for medical image analysis.  
- MDNet: Developed for medical diagnosis using images and corresponding text description.  
- Cardiac Segmentation using short-Axis MRI, segmentation of optic disc and retinal vasculature using CNN, brain tumor segmentation using random forests with features learned with fully convolutional neural network.  
