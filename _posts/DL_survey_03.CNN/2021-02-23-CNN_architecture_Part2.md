---
title:  "Deep Learning survey_5.CNN_acthitectures_Part.2(GoogLeNet, ResNet, DenseNet)"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the fifth part (popular CNN architecture section) of summary of a survey paper.
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

The basic building components (convolution and pooling) are almost the same across these architectures. However, some topological differences are observed in the modern deep learning architectures.



#### 3.2.6. GoogLeNet (2014)  

- GoogLeNet, the winner of ILSVRC 2014, was a model proposed by Christian Szegedy of Google with the objective of reducing computation complexity compared to the traditional CNN.  
- The proposed method was to incorporate *Inception Layers* that had variable receptive fields, which were created by different kernel sizes.
- These receptive fields created operations that captured sparse correlation patterns in the new feature map stack.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure6.png" style="width:60%">
  <figcaption>Fig.6 - Inception layer: Naive version.</figcaption>
</p>

The initial concept of the Inception layer can be seen in Figure 6. GoogLeNet improved state-of-the-art recognition accuracy using a stack of Inception layers, seen in Figure 7.

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure7.png" style="width:60%">
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
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure8.png" style="width:30%">
  <figcaption>Fig.8 - Basic diagram of the Residual block.</figcaption>
</p>

- The basic block diagram of the ResNet architecture is shown in Figure8.  
- The output of a residual layer can be defined based on the outputs of (l-1)<sup>th</sup> which comes from the previous layer defined as x<sub>l-1</sub>.  
- F(x<sub>l-1</sub>) is the output after performing various operations (e.g., convolution, Batch Normalization, activation function), the final output of residual unit is x<sub>l</sub> which can be defined with the following equation:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/equation6.png" style="width:30%">
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
