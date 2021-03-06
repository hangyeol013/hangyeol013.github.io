---
title:  "Deep Learning survey_4.CNN_acthitectures_Part.1(LeNet, AlexNet, VGGNet)"
search: true
categories:
  - Deep learning
classes: wide
summary: This post is the fourth part (popular CNN architecture section) of summary of a survey paper.
last_modified_at: 2021-02-23T08:06:00-05:00
---


This post is the fourth part (popular CNN architecture section) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  
This part is the basics of CNNs so, if you have already known them, you can skip this part.


### 3.2. Popular CNN Architectures

In this section, several popular state-of-the-art CNN architectures will be examined. In general, most deep convolutional neural networks are made of a key set of basic layers, including the `convolution layer`, the `sub-sampling layer`, `dense layers`, and the `soft-max layer`. The architectures typically consist of stacks of several convolutional layers and max-pooling layers followed by a fully connected and SoftMax layers at the end.  
- LeNet, **AlexNet**, **VGG Net**, NiN  
- **DenseNet**, **FractalNet**, **GoogLeNet**, Inception units, Residual Networks  
(Bold: the most popular architectures because of their state-of-the-art performance)  
(* Fractal Net is an alternative of ResNet model)

The basic building components (convolution and pooling) are almost the same across these architectures. However, some topological differences are observed in the modern deep learning architectures.


#### 3.2.1. LeNet (1998)  

- Paper: [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791) by Yann LeCun.  
- `Limited computation capability` and `memory capacity` made the algorithm difficult to implement until about 2010.  
- LeCun proposed CNNs with the `back-propagation algorithm` and experimented on handwritten digit dataset to achieve state-of-the-art accuracy.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure3.png" style="width:50%">
  <figcaption>Fig.3 - The architecture of LeNet-5.</figcaption>
</p>

- They used average pooling layer instead of max-pooling layer.  
- Activation function was Tanh function, zero-padding with a stride of one.  
- Trained using Maximum Likelihood Estimation (MLE) and Mean Squared Error (MSE) loss.  
- The input images for LeNet were normalised such that their values remain within the range [-0.1, 1.175], which made the mean 0 and the variance roughly 1. It was argued that normalisation accelerated the network's learning.  
- LeNet-1, LeNet-4, LeNet-5


**Several Limitations**

1. The networks were `small` and therefore had `limited applications` where they worked correctly.  
2. The networks only worked with `single-channel` (black and white images), which also limited its applications.  
3. Most modern adaptations of this model implement a max-pooling operation as opposed to an `average pooling` operation as it allows for more rapid convergence due to the larger gradients returned during back-propagation.  
4. Most modern adaptations implement a ReLu activation as opposed to `tanh` and `sigmoid` as ReLu usually leads to higher classification accuracies.  

<br>

From this part, I referred a post from [Review of LeNet](https://towardsdatascience.com/review-of-lenet-5-how-to-design-the-architecture-of-cnn-8ee92ff760ac)  

##### Globally trainable system

###### What's a globally trainable system?  

From the perspective of back-propagation, if all the modules are differentiable and the connections between the modules are also differential, in other words, the back propagation of gradients can go back from the loss function at the end to the input, it is a globally trainable system. Sometimes, we also name it as `end-to-end` solution for machine learning problem.  

**+)What is the advantage of a globally trainable system?**  
- We can't use the loss function of classification to optimize the performance of localization if the two modules (classification and localization) are separate.  
- In traditional solution, we often feed the machine learning model with the hand-designed features, but with globally trainable system, we can let the data tell us which are the most important features for a certain task.  


##### Design the CNN with knowledge  

In order to get self-learned features from neural network, we have to design a good architecture for the neural network.  Yann LuCun indicated in his paper that *'No learning technique can succeed without a minimal amount of prior knowledge about the task. ... a good way to incorporate with knowledge is to tailor its architecture to the task'*.

**Coonvolutional Kernels)**  
In 1962, Hubel and Wiesel revealed that locally-sensitive, orientation-selective neuron in the cat's visual systems. With the local connections, the neuron can learn `some basic visual features`, which could be reused or grouped to form new features in the following neuron. A convolution kernel can perfectly realize this `receptive field`.  

Then how can we overcome some common `difficulties of image classification`: shift, scale and distortion invariance. Let's first check how human being realize image classification.  
 1. Scan the image with `some visual pattern` to find some features.  
 2. Find the `relation between features`.  
 3. Search `the pattern of relation in the pattern database in our brain`.  
 4. Find `the most similar one`.  

**To realize**  
**The step 1**, we can fix `the weights of convolution kernel` for the same feature map and generate several feature maps.  
**The step 2**, we consider the exact position of the feature are less important than the **relative position of the feature** to other features. So we can progressively reduce the spatial resolution (sub-sampling).
(However, we lose information when reducing the image resolution. That's why we need to increase the number of feature maps to keep the useful information as much as possible)  

With all these knowledge, we have general principle to design a CNN.  
1. Use convolutional kernel  
2. Shared weights  
3. Sub-sampling and increasing the number of feature maps  


<br>


#### 3.2.2. AlexNet (2012)  

- Paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf).  
- In 2012, Alex Krizhevesky proposed a deeper and wider CNN model compared to LeNet and won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012.
- It was a significant breakthrough in the field of machine learning and computer vision for visual recognition and classification tasks and is the point in history where interest in deep learning increased rapidly.  
- The highlights of this paper: `Breakthrough` in Deep Learning using CNN for image classification, `Multi-GPUs`, `Use ReLU`, `Use Dropout`.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure4.png" style="width:80%">
  <figcaption>Fig.4 - The architecture of AlexNet.</figcaption>
</p>

- AlexNet contains five convolutional and three fully-connected layers.  
- The output of the last fully-connected layer is sent to a 1000-way softmax layer which corresponds to 1000 class labels in the ImageNet dataset.  

**ReLU nonlinearity**  

Before AlexNet, `sigmoid` and `tanh` were usually used as activations which are `saturating nonlinearities`. AlexNet uses `Rectified Linear Units (ReLU)` activations which are `non-saturating nonlinearity`.  
The benefits of ReLU are:  
1. Avoid vanishing gradients for positive values.  
2. More computationally efficient to compute than sigmoid and tanh.  
3. Better convergence performance than sigmoid and tanh.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure15.png" style="width:50%">
  <figcaption>Fig.5 - A four-layer convolutional neural network with ReLUs (solid line) and network with tanh neurons (dashed line).</figcaption>
</p>


**Multi-GPUs**  

We can see that the architecture is split into two parallel parts. In AlexNet, 1.2 million training parameters are `too big to fit` the NVIDIA GTX GPU with 3 GB of memory. Therefore, the author spread the network across two GPUs. In this paper, the usage of two GPUs is due to memory limitation, not for distributed training as in current years. Nowadays, the NVIDIA GPUs are large enough to handle this tasks.


**Overlapping Pooling**  

`Pooling layers` in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighbor neurons by adjacent pooling units do not overlap. To be more precise, a pooling layer can be thought of as consisting of a grid of pooling units spaced *s* pixels apart, each summarizing a neighborhood of size *z* x *z* centered at the location of the pooling unit.  
If we set `*s*=*z*`, we obtain `traditional local pooling` as commonly employed in CNNs.  
If we set `*s*<*z*`, we obtain `overlapping pooling`.  
This is what they use throughout their network, with *s* = 2 and *z* = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%, respectively, as compared to max pooling of size 2x2 with stride 2. The author also said that they generally observe during training that models with overlapping pooling find it slightly more `difficult to overfit`.   


**Local Response Normalization**  

Local Response Normalization (LRN) is used in AlexNet to help with generalization.  

The formula of Local Response Normalization is:  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure16.png" style="width:50%">
</p>

LRN reduces the top-1 and top-5 error rates by 1.4% and 1.2%.  
In 2014, VGGNet shows that `LRN does not improve the performance` on ILSVRC dataset but leads to increased memory and computation time.  

Nowadays, batch normalization is used instead of LRN.  


##### Reduce Overfitting  

**Dropout)**  
They use `Dropout` to help reducing overfitting.  

- `Combining the predictions of many different models` is a very successful way to reduce test errors, but it appears to be `too expensive` for big neural networks that already take several days to train.  
- There is, however, a very `efficient version of model combination` that only costs about a factor of two during training.  
- `A regularization technique` which will randomly set the output of each hidden neuron to zero with the probability of *p*=0.5. Those dropped out neurons do not contribute to forward and backward passes.  
- So every time an input is presented, the neural network samples `a different architecture`, but all these architectures `share weights`.  
- This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons.  
- It is, therefore, forced to learn more robust features that are useful in conjunction with `many different random subsets of the other neurons`.  
- Traditionally, `in test time`, we will need to `multiply the outputs by p=0.5` so that the response will be the same as training time.  
In implementation, it is common to `rescale the remainder neurons`, which are not dropped out, by dividing by *(1-p)* in training time. Therefore, we `don't need to scale in test time`.  


**Data Augmentation**  
They use `Data Augmentation` to help reducing overfitting. The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations.  

AlexNet uses two forms of data augmentation, both of which allow transformed images to be produced from the original images with very little computation, so the transformed images do not need to be stored on disk.  

1. `Translations and horizontal reflections`)  
- This technique increases the size of training set.  
2. Altering the `intensities of RGB channels`  
- Perform PCA on the set of RGB pixel values throughout the training set. Then use the eigenvalues and eigenvectors to manipulate the pixel intensities. Eigenvalues are selected once for entire pixels of an particular image.  


**Ohter details**  

- Batch size: 128  
- Momentum: 0.9  
- Weight Decay: 0.0005  
- Initialize the weights in each layer from a zero-mean Gaussian distribution with std 0.01  
- Bias: Initialize 1 for 2nd, 4th, 5th conv layers and fully-connected layers. (Initialize 0 for remaining layers)  
- Learning rate: 0.01 (Diving by 10 when validation error stopped improving)  

Train roughly 90 cycles with 1.2 million training images, which took 5 to 6 days on two NVIDIA GTX 580 3GB GPUs.  

<br>


#### 3.2.3. ZFNet / Clarifai (2013)  

- In 2013, Matthew Zeiler and Rob Fergue won the 2013 ILSVRC with a CNN architecture which was an extension of AlexNet.  
- As CNNs are expensive computationally, an optimum use of parameters is needed from a model complexity point of view.  
- ZFNet uses 7x7 kernels instead of 11x11 kernels to significantly reduce the number of weights.  
- This reduces the number of network parameters dramatically and improves overall recognition accuracy.  

<br>

#### 3.2.4. Network in Network (NiN)  

- This model is slightly different from the previous models where a couple of new concepts are introduced.  
<br>
**1) multilayer perception convolution**
- convolutions are performed with 1x1 filter that help to add more nonlinearity in the models.  
- This helps to increase the depth of the network, which can then be regularized with dropout.  
- This concept is used often in the bottleneck layer of a deep learning model.  
<br>
**2) Global Average Pooling (GAP) as an alternative of fully connected layers.**  
- This helps to reduce the number of network parameters significantly.  
- By applying GAP on a large feature map, we can generate a final low dimensional feature vector without reducing the dimension of the feature maps.  

<br>

#### 3.2.5. VGGNet (2014)  

- Paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf).  
- I also referred to [a VGGNet review post](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)  
- The Visual Geometry Group (VGG), was the runner-up of the 2014 ILSVRC.  
- The main contribution of this work is that it shows that the depth of a network is a critical component to achieve better recognition or classification accuracy in CNNs.  
- The first year that there are deep learning models obtaining the error rate under 10%.  
- The most important is that there are many other models built on top of VGGNet or based on the 3x3 conv idea of VGGNet.  
- Three VGG-E models, VGG-11, VGG-16 and VGG-19 were proposed the model had 11, 16 and 19 layers respectively.  
- All versions of the VGG-E models ended the same with three fully connected layers.  
- The highlights of this paper: `The use of 3x3 Filters` instead of large-size filters (such as 11x11, 7x7), `VGG-16 and VGG-19 based on ablation study`, `Multi-scale training`, `Multi-scale testing`, `Dense Testing`, `Model fusion`.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure5.png" style="width:60%">
  <figcaption>Fig.5 - The basic building block of VGG network.</figcaption>
</p>


**ImageNet)**
- A dataset of over 15 millions labeled high-resolution images with around 22,000 categories.  
- ILSVRC uses a subset of ImageNet of around 1000 images in each of 1000 categories.  
- In all, there are roughly 1.3 million training images, 50,000 validation images and 100,000 testing images.  

###### The use of 3x3 Filters  

- By using 2 layers of 3x3 filters, it actually have already covered 5x5 area and by using 3 layers of 3x3 filters, it actually have already covered 7x7 effective area.  
- Thus, large-size filters such as 11x11 in AlexNet and 7x7 in ZFNet indeed are not needed.  

- The number of parameters are also **fewer**.  
**1 layer of 11x11 filter**, # of parameters = 11 x 11 = 121
**5 layers of 3x3 filter**, # of parameters = 3 x 3 x 5 = 45
-> **Number of parameters is reduced by 63%**  

**1 layer of 5x5 filter**, # of parameters = 5 x 5 = 25  
**2 layers of 3x3 filter**, # of parameters = 3 x 3 + 3 x 3 = 18  
-> **Number of parameters is reduced by 28%**  

With fewer parameters to be learnt, it is better for faster convergence, and reduced overfitting problem.  


###### VGG-16 and VGG-19 Based on Ablation Study  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure17.png" style="width:60%">
  <figcaption>Fig.6 - Different VGG Layer Structures Using Single Scale Evaluation.</figcaption>
</p>

To obtain the optimum deep learning layer structure, ablation study has been done as shown in the above figure.  
- The one with additional local response normalization operation suggested by AlexNet doesn't improve which means LRN is not useful.  
- The additional three 1x1 conv layers help the classification accuracy. 1x1 conv actually helps to increase non-linearity of the decision function. Without changing the dimensions of input and output, 1x1 conv is doing the projection mapping in the same high dimensionality.  
- The deep learning network is not improving by just adding number of layers (From VGG-16 to VGG-19)
- We can observe that VGG-16 and VGG-19 start converging and the accuracy improvement is slowing down.  


###### Multi-scale Training  

As object has different scale within the image, if we only train the network at the same scale, we might miss the detection or have the wrong classification for the objects with other scales. To tackle this, authors propose multi-scale training.  

For **single-scale training**,  
an image is scaled with smaller-size equal to 256 or 384 (i.e. S=256 or 384). Since the network accepts 224x224 input images only, the scaled image will be cropped to 224x224 like the below figure.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure18.png" style="width:60%">
  <figcaption>Fig.6 - Single-scale training with S=256 and S=384.</figcaption>
</p>

For **multi-scale training**,  
an image is scaled with smaller-size equal to a range from 256 to 512 (i.e. S=[256;512]), then cropped to 224x224. Therefore, with a range of S, we are inputting different scaled objects into the network for training.  
By using multi-scale training, we can imagine that it is more accurate for test image objects with different object sizes.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure19.png" style="width:60%">
  <figcaption>Fig.7 - Multi-scale Training Results.</figcaption>
</p>


###### Multi-scale Testing  

Similar to multi-scale training, multi-scale testing can also reduce the error rate since we do not know the size of object in the test image. If we scale the test image to different sizes, we can increase the chance of correct classification.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure20.png" style="width:60%">
  <figcaption>Fig.8 - Multi-scale Testing Results.</figcaption>
</p>

- By using multi-scale testing but single-scale training, error rate is reduced compared to single-scale training & single-scale testing.  
- By using both multi-scale training and testing, error rate is reduced compared to only multi-scale testing.  


###### Dense (Convolutionalized) Testing  

Multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, while in the case of dense evaluation the padding for the same crop naturally comes form the neighboring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured.  

Using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them. We hypothesize that this is due to a different treatment of convolution boundary conditions.  

<p>
  <img src="/assets/images/blog/DL_survey_03.CNN/Figure21.png" style="width:60%">
  <figcaption>Fig.9 - ConvNet evaluation techniques comparison.</figcaption>
</p>

<br>

In the next post, I will discuss about the popular CNN architectures part.2 (GoogLeNet, ResNet, DenseNet).
