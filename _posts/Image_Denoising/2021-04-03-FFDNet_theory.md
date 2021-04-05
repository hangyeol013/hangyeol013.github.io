---
title:  "FFDNet paper_Review_Theory"
search: true
categories:
  - Image denoising
date: April 02, 2021
summary: This post is a summary of FFDNet (Fast and Flexible Denoising CNN) paper.
toc: true
toc_sticky: true
header:
  teaser: /assets/images/thumbnails/thumb_basic.jpg
tags:
  - Deep Learning
  - Image denoising
last_modified_at: 2021-04-05T08:06:00-05:00
---


This post is a summary of a FFDNet (Fast and Flexible Denoising CNN) paper  
[FFDNet: Toward a Fast and Flexible Soultion for CNN based Image Denoising](https://arxiv.org/pdf/1710.04026.pdf).  

<br>


### 1. Introduction  

In general, image denoising methods can be grouped into two major categories, `model-based methods` and `discriminative learning based methods`.  

**Model-based methods**  
- BM3D, WNNM  
- Flexible in handling denoising problems with `various noise levels`  
- Optimization algorithms are generally `time-consuming`  
- `Cannot be directly used to remove spatially variant noise`  
- Usually employ `hand-crafted image priors`, which may not be strong enough to characterize complex image structures  

*What is model-based neural networks?*  
The use of MBNNs allows a network to be constructed in which `Supervisor's knowledge` of the task to be performed is `used to specify partially the roles of some hidden units`, or of whole hidden layers, in advance. Thus the supervisor's knowledge of which components of the training data are significant for the task is incorporated into the network geometry and connection weighting functions, serving as a constraint on the state space searched.  


**Discriminative learning-based methods**  
- MLP, DnCNN  
- Aim to `learn the underlying image prior`  
- `fast inference` from a training set of degraded and ground-truth image pairs  
- `Limited in flexibility` (lack flexibility to deal with spatially variant noise)  
- Learned model is usually `tailored to specific noise level`  
- Even DnCNN-B `cannot generalize well to real noisy images` (work only in the present range [0, 55])  



**Difference between DnCNN and FFDNet**  

**DnCNN)**  
- Formulated as x = F(y; &theta;<sub>&sigma;</sub>)
- `The parameter` &theta;<sub>&sigma;</sub> `vary with the change of noise level` &sigma;  
(&rarr; In case of DnCNN-S. DnCNN-B has the parameters invariant to the change of noise level?)  

**FFDNet)**  
- Formulated as x = F(y,M; &theta;)
- M is a `noise level maps`  
- The `noise level map` is modeled `as an input` and the `model parameters` &theta; are `invariant to noise level`  
- Thus, FFDNet provides a `flexible` way `to handle different noise levels` with a single network  

By introducing a noise level map as input, it is natural to expect that the model performs well when the noise level map matches the ground-truth one of noisy input.  

Furthermore, the noise level map should also play `the role of controlling the trade-off between noise reduction and detail preservation`.  
- large noise level: noise reduction  
- small noise level: detail preservation  

A larger noise level can smooth out the detail so that `heavy visual quality degradation` may be engendered. The authors adopt a method of `orthogonal initialization` on conventional filters to alleviate this problem.  

*Orthogonal initialization: a simple yet relatively effective way of combatting exploding and vanishing gradients.

Besides, the proposed FFDNet works on downsampled sub-images, which largely accelerates the training and testing speed, and enlarges the receptive field as well.  

The main contribution of their work is summarized as follows:  
1. By taking a `tunable noise level map as input`, a single FFDNet is able to deal with `noise on different levels`, as well as `spatially variant noise`.  
2. They highlight the importance to guarantee the role of the noise level map in `controlling the trade-off` between noise reduction and detail preservation.  
3. FFDNet exhibits perceptually appealing results, demonstrating its potential for practical image denoising.  

<br>

### 2. Related Work    


#### A. MAP Inference Guided Discriminative Learning  

Here, I reviewed about MAP (Maximum A posteriori) comparing with MLE (Maximum Likelihood Estimation). I referred to the content from Shota Horii's post, you can access and read more in detail on his post:  
[A Gentle Introduction to Maximum Likelihood Estimation and Maximum A Posteriori Estimation](https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-and-maximum-a-posteriori-estimation-d7c318f9d22d)  

`MLE` and `MAP` estimation are method of `estimating parameters of statistical models`.  

With the assumption that probability &theta; follows binomial distribution, The formula of the probability is below:  

<p>
  <img src="/assets/images/blog/Image_Denoising/Equation2.png" style="width:70%">
</p>

I took this equation from his post, so it's expressed with 'k wins out of n matches', which is an example in his post.  
This simplification is the statistical modelling of the example, and &theta; is `the parameter to be estimated with MLE and MAP`.  

--------------------------------

**MLE (Maximum Likelihood Estimation)**  

Likelihood: *P(D|&theta;)*  
'MLE' aims to solve 'What is `the exact value of` &theta; which `maximize the likelihood` *P(D|&theta;)*?'  
The value of &theta; maximizing the likelihood can be obtained by having derivative of the likelihood function with respect to &theta;, and setting it to zero.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Equation3.png" style="width:70%">
</p>

By solving this, &theta; = 0,1 or k/n.

--------------------------------

**MAP (Maximum A Posteriori Estimation)**  

'MLE' is powerful when you have enough data. However, it doesn't work well when `observed data size is small`. However, when you have `a prior knowledge`, it can be helpful to estimate the parameters. This prior knowledge is called `prior probability`, P(&theta;).  

Then, the updated probability of &theta; given D (observed data) is expressed as *P(&theta;\D)* and called the `posterior probability`.  

Now, we want to know the best guess of &theta; considering both our prior knowledge and the observed data. It means `maximizing the posterior probability`, *P(&theta;\D)* and it's the MAP estimation.


<p>
  <img src="/assets/images/blog/Image_Denoising/Equation4.png" style="width:20%">
</p>
Here, we can calculate *P(&theta;|D)* using `Bayes' theorem` below.  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation5.png" style="width:25%">
</p>
P(D) is independent to the value of &theta; and since we're only interested in finding &theta; maximizing *P(&theta;|D)*, we can ignore *P(D)* in the maximization.  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation6.png" style="width:50%">
</p>

Intrinsically, we can use any formulas describing probability distribution as *P(&theta;)* to express the prior knowledge well. However, for the computational simplicity, specific probability distributions are used corresponding to the probability distribution of likelihood. It's called `conjugate prior distribution`.  

Since the conjugate prior of binomial distribution is `Beta distribution`, we use Beta distribution to express *P(&theta;)* here.  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation7.png" style="width:40%">
</p>
Where, &alpha; and &beta; are called hyperparameter, which cannot be determined by data. Rather we set them subjectively to express our prior knowledge well.  

So, by now we have all the components to calculate *P(D|&theta;)P(&theta;)* to maximize.  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation8.png" style="width:60%">
</p>

As same as MLE, we can get &theta; maximizing this by having derivative of the this function with respect to &theta;, and setting it to zero.  

--------------------------------

Instead of first learning the prior and then performing the inference, this category of methods *aims to learn the prior parameters along with a compact unrolled inference through minimizing a loss function*.  

`MAP inference guided discriminative learning` usually requires `much fewer inference steps`, and is very efficient in image denoising. It also has `clear interpretability` because the discriminative architecture is derived from optimization algorithms.  

However, the learned priors and inference procedure are `limited by the form of MAP model`, and generally `perform inferior` to the state-of-the-art CNN-based denoisers.  

- Discriminative Markov random field (MRF) model, CSF framework, TNRD model, network based on GCRF (Gaussian Conditional Random Field) inference  


#### B. Plain Discriminative Learning  

Instead of modeling image priors explicitly, the `plain discriminative learning methods` learn a `direct mapping function to model image prior` implicitly.  

Plain discriminative learning has shown `better performance` than MAP inference guided discriminative learning. However, existing discriminative learning methods have to learn `multiple models for handling images with different noise levels`, and are `incapable to deal with spatially variant noise`.  

It remains an unaddressed issue to develop a single discriminative denoising model which can handle noise of different levels, even spatially variant noise, in a speed even faster than BM3D.  

- MLP, DnCNN, RBDN (recursively branched deconvolutional network), MemNet  

<br>

### 3. Proposed Fast and Flexible Discriminative CNN Denoiser  

They take mainly three strategies to make the denoising network flexible to noise level, be efficient, and robustly control the trade-off between noise reduction and detail preservation:  
1. Flexible: Take a `tunable noise level M as input`  
2. Efficiency: Introduce a `reversible downsampling operator` to reshape the input image (W x H x C &rarr; W/2 x H/2 x 4C)  
3. Robustness: Adopt the `orthogonal initialization` method to the convolution filters  


#### A. Network Architecture  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure8.png" style="width:100%">
  <figcaption>
  Fig.1 - The architecture of the proposed FFDNet for image denoising. The input image is reshaped to four sub-images, which are then input to the CNN together with a noise level map. The final output is reconstructed by the four denoised sub-images
  </figcaption>
</p>

- The first layer is a `reversible downsampling operator` (downsampling factor 2) which reshapes a noisy image *y* `into four downsampled sub-images`.  
- They further concatenate a tunable `noise level map M` with the downsampled sub-images to form a tensor *y&#771;* of size `W/2 x H/2 x (4C+1)` as the input to CNN.  
(For spatially invariant AWGN with noise level &sigma;, M is uniform map with all elements being &sigma;.)  
- Zero-padding for each layer  
- After the last convolution layer, an `upscaling operation` (sub-pixel convolution layer) is applied as the `reverse operator of the downsampling operator` applied in the input stage.  
- Different from DnCNN, FFDNet does not predict the noise.  
- filter size: 3 x 3  
- 15 layers, 64 channels of feature maps (grayscale image), 12 layers, 96 channels of feature maps (color image)  

*Why did they use different settings for grayscale and color image?*  
1. Since there are `high dependencies among the R, G, B channels`, using a smaller number of convolution layers encourages the model to exploit the `inter-channel dependency`.  
2. Color image has `more channels` as input, and hence `more feature` (i.e., more channels of feature map) `is required` (experimentally).  

&rarr; Using different settings for color images, FFDNet can bring an average gain of 0.15dB by PSNR on different noise levels.  


#### B. Noise Level Map  

Let's first revisit the model-based image denoising methods to analyze *why they are flexible in handling noises at different levels*.  

Most of the model-based denoising methods aim to solve the following problem.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Equation9.png" style="width:50%">
</p>

- `||y-x|| term`: Data fidelity term with noise level &sigma;  
- &Phi;(x): the regularization term associated with image prior  
- &lambda;: controls the balance between the data fidelity and regularization term  

In practice &lambda; governs `the compromise between noise reduction and detail preservation`.  
- When it is `too small`: much noise will remain  
- when it is `too big`: details will be smoothed out along with suppressing noise  

With some optimization algorithms, the solution of above equation actually defines a implicit function given by  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation10.png" style="width:20%">
</p>

&lambda; can be absorbed into &sigma;. In this sense, *setting noise level &sigma; also plays the role of setting &lambda; to control the trade-off between noise reduction and detail preservation*. In a word, model-based methods are flexible in handling images with various noise levels by simply specifying &sigma; in above equation.  

However, since the inputs *y* and &sigma; have different dimensions, it is not easy to directly feed them into CNN. They resolve the dimensionality mismatching problem by stretching the noise level &sigma; into a noise level map *M*. As a result, above equation can be further rewritten as  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation11.png" style="width:20%">
</p>
*M* can be extended to degradation maps with multiple channels for more general noise models such as the multivariate (3D) Gaussian noise model.  


#### C. Denoising on Sub-images  

`Efficiency` is another crucial issue for practical CNN-based denoising. One straightforward idea is to reduce the depth and number of filters. However, such a strategy will sacrifice much the modeling capacity and receptive field of CNN.  

The authors introduce `a reversible downsampling layer` to reshape the input image into a set of small sub-images. Here `the downsampling factor is set to 2` since it can largely improve the speed without reducing modeling capacity. The CNN is deployed on the sub-images, and finally `a sub-pixel convolution layer` is adopted to reverse the downsampling process.  

Denoising on downsampled sub-images can also effectively `expand the receptive field` which in turn leads to a moderate network depth. What is more, the introduction of subsampling and sub-pixel convolution is effective in `reducing the memory burden`. *As a result, by performing denoising on sub-images, FFDNet significantly improves efficiency while maintaining denoising performance*.  


#### D. Examining the Role of Noise Level Map  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure9.png" style="width:80%">
  <figcaption>
  Fig.2 - An example to show the importance of guaranteeing the role of noise level map in controlling the trade-off between noise reduction and detail preservation. The input is a noisy image with noise level 25. (a) Result without visual artifacts by matches noise level 25. (b) Result without visual artifacts by mismatches noise level 60. (c) Result with visual artifacts by mismatches noise level 60.
  </figcaption>
</p>

By training the model with abundant data units (*y,M;x*), where *M* is exactly the noise level map of *y*, the model is expected to perform well when the noise level matches the ground-truth one. On the other hand, one may take advantage of the role of &lambda; to control the trade-off between noise reduction and detail preservation.  

Unfortunately, the use of both *M* and *y* as input also increases the `difficulty to train` the model. The model may give rise to visual artifacts especially when the input noise level is much higher than the ground-truth one (Fig.2(c)), which indicates *M* fails to control the trade-off between noise reduction and detail preservation.  

*One possible solution to avoid this is to regularize the convolution filters*. As a widely-used regularization method, `orthogonal regularization` has proven to be effective in eliminating the correlation between convolution filters, facilitating gradient propagation and improving the compactness of the leaned model.  


#### E. FFDNet vs. a Single Blind Model  

It is of significant importance to clarify the differences between a single model for blind and non-blind Gaussian denoising.  
1. The `generalization ability` is different.  
- Although the blind model performs favorably for synthetic AWGN removal without knowing the noise level, it does not generalize well to real noisy images whose noises are much more complex than AWGN.  
(a model trained for AWGN removal is not expected to be still effective for Poisson noise removal)  
- By contrast, the non-blind FFDNet model can be viewed as multiple denoisers, each of which is anchored with a noise level.  
- Accordingly, it has the ability to control the trade-off between noise removal and detail preservation which in turn facilitates the removal of real noise to some extent.  
2. The `performance for AWGN removal` is different.  
- The non-blind model with noise level map has moderately better performance for AWGN removal than the blind one.  
3. The `application range` is different.  
- The non-blind model can be easily plugged into variable splitting algorithms to solve various image restoration tasks. (image deblurring, SISR, image inpainting)  
- However, the blind model does not have this merit.  



#### F. Residual vs. Non-residual Learning of Plain CNN  

`In DnCNN`, it has been pointed out that the integration of `residual learning` for plain CNN and `batch normalization` is beneficial to the removal of AWGN as it eases the training and tends to deliver better performance. The main reason is that the residual (noise) output follows a Gaussian distribution which facilitates the Gaussian normalization step of batch normalization. The denoising network gains most from such a task-specific merit especially when a single noise level is considered.  

`In FFDNet`, the authors consider a wide range of noise level and introduce a noise level map as input. According to their experiments, *with BN, while the residual learning enjoys a faster convergence than non-residual learning, their final performances after fine-tuning are almost exactly the same*. In fact, when a network is moderately deep (e.g., less than 20), it is feasible to train a plain network without the residual learning strategy by using advanced CNN training and design techniques (ReLU, BN, Adam).  

`For simplicity`, they do not use residual learning for network design.  



#### G. Un-clipping vs. Clipping of Noisy Images for Training  

In the AWGN denoising literature, there exist two widely-used settings, i.e., `un-clipping` and `clipping`, of synthetic noisy image to evaluate the performance of denoising methods. The main different between the two settings lies in `whether the noisy image is clipped into the range of 0-255` (or more precisely, `quantized into 8-bit format` after adding the noise)  

The `un-clipping setting` serves `an ideal test bed` for evaluating the denoising methods. This is because most denoising methods assume the noise is ideal AWGN, and the clipping of noisy input would make the noise characteristics deviate from being AWGN. Thus, unless otherwise specified, FFDNet in this work refers to the model trained on images without clipping or quantization.  

On the other hand, since `real noisy images` are always `integer-valued and range-limited`, it has been argued that the clipping setting of noisy image makes the data more realistic. However, when the noise level is high, the noise will be not zero-mean any more due to clipping effects. This in turn will lead to unreliable denoiser for plugging into the variable splitting algorithms to solve other image restoration problems.  

To thoroughly evaluate the proposed method, the authors also train an FFDNet model with clipping setting of noisy image, namely FFDNet-Clip. During training and testing of FFDNet-Clip, the noisy images are quantized to 8-bit format.  



### 4. Experiments  


#### A. Dataset Generation and Network Training  

To train the FFDNet model, we need to prepare a training dataset of input-output pairs {(*y<sub>i</sub>,M<sub>i</sub>;x<sub>i</sub>*)}<sup>N</sup><sub>i=1</sub>.  
- *y<sub>i</sub>*: obtained by adding AWGN to latent image *x<sub>i</sub>*  
- *M<sub>i</sub>* is the noise level map  


*The reason to use AWGN to generate the training dataset is two-fold.*  
1. AWGN is a natural choice when there is no specific prior information on noise source.  
2. Real-world noise can be approximated as locally AWGN.  

More specifically, FFDNet model is trained on the noisy images *y<sub>i</sub>* = *x<sub>i</sub>* + *v<sub>i</sub>* `without quantization to 8-bit integer values`.  


**Dataset)**  
- 400 (BSD images), 400 images (the validation set of ImageNet), 4,744 images (Waterloo Exploration Dataset).  
- In each epoch, randomly crop N = 128 x 8,000  
- Grayscale: 70 x 70 (patch size), 64 (# of channels)  
- Color: 50 x 50 (patch size), 96 (# of channels)  
- Noise level: &sigma; &isin; [0, 75]  
- For each noisy patch, the noise level map is uniform.  
- Adam optimization (default settings)  
- Loss function:  
<p>
  <img src="/assets/images/blog/Image_Denoising/Equation12.png" style="width:50%">
</p>
- Leaning rate: 1e-3 to 1e-4 (when the training error stops decreasing)  
- mini batch size: 128  
- Data augmentation (rotation & flip)  


#### B. Experiments on AWGN Removal  

In this section, we'll examine FFDNet on noisy images corrupted by spatially invariant AWGN.  
- BM3D, WNNM: model-based methods based on nonlocal self-similarity prior  
- TNRD, MLP, DnCNN: Discriminative learning based methods  

<p>
  <img src="/assets/images/blog/Image_Denoising/Table4.png" style="width:100%">
  <figcaption>
  Table.1 - The average PSNR (dB) results of different methods on BSD68 with noise levels 15, 25, 35, 50 and 75.
  </figcaption>
</p>

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure10.png" style="width:100%">
  <figcaption>
  Fig.3 - Denoising results on image '102061' from the BSD68 dataset with noise level 50 by different methods.
  </figcaption>
</p>

1. FFDNet outperforms other methods for a wide range of noise level.  
2. FFDNet is slightly inferior to DnCNN when the noise level is low (e.g., &sigma; &#8804; 25), but gradually outperforms DnCNN with the increase of noise level. This is may be resulted from the trade-off between receptive field size and modeling capacity.  
(FFDNet has a larger receptive field than DnCNN, thus favoring for removing strong noise, while DnCNN has better modeling capacity which is beneficial for denoising images with lower noise level)  
3. On images which have a rich amount of repetitive structures, FFDNet is inferior to WNNM.  



**Color image denoising**  

<p>
  <img src="/assets/images/blog/Image_Denoising/Table5.png" style="width:80%">
  <figcaption>
  Table.2 - The average PSNR (dB) results of CBM3D, CDnCNN and FFDNet on CBS68, Kodak24 and McMaster datasets with noise levels 15, 25, 35, 50 and 75.
  </figcaption>
</p>

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure11.png" style="width:80%">
  <figcaption>
  Fig.4 - Color image denoising results by CBM3D, CDnCNN and FFDNet on noise level &sigma; = 50.
  </figcaption>
</p>

- FFDNet consistently outperforms CBM3D on different noise levels in terms of both quantitative and qualitative evaluation, and has competing performance with CDnCNN.  


#### C. Experiments on Spatially Variant AWGN Removal  

To synthesize `spatially variant AWGN`, they first generate an AWGN image *v<sub>1</sub>* with unit standard deviation and a noise level map *M* of the same size. Then, element-wise multiplication is applied on *v<sub>1</sub>* and *M* to produce the spatially variant AWGN, i.e., *v* = *v<sub>1</sub>* &#9737; *M*.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure12.png" style="width:80%">
  <figcaption>
  Fig.5 - Examples of FFDNet on removing spatially variant AWGN. (a) Noisy image (20.55dB) with spatially variant AWGN. (b) Ground-truth noise level map and corresponding denoised image (30.08dB) by FFDNet, (c) uniform noise level map constructed by using the mean value of ground-truth noise level map and corresponding denoised image (27.45dB) by FFDNet.
  </figcaption>
</p>

- `FFDNet with non-uniform noise level map` is flexible and powerful to remove spatially variant AWGN.  
- In contrast, `FFDNet with uniform noise level map` would fail to remove strong noise at the region with higher noise level while smoothing out the details at the region with lower noise level.  


#### D. Experiments on Noise Level Sensitivity  

*In practical applications, the noise level map may not be accurately estimated from the noisy observation, and mismatch between the input and real noise levels is inevitable*. A practical denoiser should tolerate certain mismatch of noise levels. In this subsection, the authors evaluate FFDNet by varying different input noise levels for a given ground-truth noise level.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure14.png" style="width:80%">
  <figcaption>
  Fig.6 - Noise level sensitivity curves of BM3D, DnCNN and FFDNet. The image noise level on x-axis means ground-truth noise level. The averaged PSNR results are evaluated on BSD68.
  </figcaption>
</p>

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure13.png" style="width:100%">
  <figcaption>
  Fig.7 - Visual comparisons between FFDNet and BM3D/CBM3D by setting different input noise levels to denoise a noisy image. (a) From top to bottom: ground-truth image, four clean zoom-in regions, and the corresponding noisy regions (AWGN, noise level 15). (b) From top to bottom: denoising results by BM3D with input noise levels 5, 10, 15, 20, 50 and 75, respectively. (c) Results by FFDNet with the same settings as in (b). (d) From top to bottom: ground-truth image, four clean zoom-in regions, and the corresponding noisy regions (AWGN, noise level 25). (e) From top to bottom: denoising results by CBM3D with input noise levels 10, 20, 25, 30, 45 and 60, respectively. (f) Results by FFDNet with the same settings as in (e).
  </figcaption>
</p>

- On all noise levels, FFDNet achieves similar denoising results to BM3D and DnCNN.
- With the fixed input noise level, the PSNR value tends to stay the same when the ground-truth noise level is lower, and begins to decrease when the ground-truth noise level is higher.  
- The best visual quality is obtained when the input noise level matches the ground-truth one.  
- Using a higher input noise level can generally produce better visual results than using a lower one. In addition, there is no much visual difference when the input noise level is little higher than the ground-truth one.  

*When the ground-truth noise level is unknown, it is more preferable to set a larger input noise level than a lower one to remove noise with better perceptual quality*.  


#### E. Experiments on Real Noisy Images  

In this subsection, real noisy images are used to further assess the practicability of FFDNet. FFDNet focuses on non-blind denoising and assumes the noise level map is known.  

Instead of adopting any noise level estimation methods, the authors adopt `an interactive strategy` to handle real noisy images.  
1. They employ a set of typical input noise levels to produce multiple outputs, and select the one which has best trade-off between noise reduction and detail preservation.  
2. They first sample several typical regions of distinct colors. For each typical region, they apply different noise levels with an interval of 5, and choose the best noise level by observing the denoising results.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure15.png" style="width:90%">
  <figcaption>
  Fig.8 - Grayscale image denoising results by different methods on real noisy images From top to bottom, (1) noisy image (2) denoised image by Noise Clinic, (3) denoised image by BM3D, (4) denoised image by DnCNN, (5) denoised by DnCNN-B, (6) denoised image by FFDNet &sigma; = 15 (10 for DnCNN).
  </figcaption>
</p>

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure16.png" style="width:90%">
  <figcaption>
  Fig.9 - Grayscale image denoising results by different methods on real noisy images From top to bottom, (1) noisy image (2) denoised image by Noise Clinic, (3) denoised image by BM3D, (4) denoised image by DnCNN, (5) denoised by DnCNN-B, (6) denoised image by FFDNet &sigma; = 20 (10 for DnCNN).
  </figcaption>
</p>

- While the non-blind DnCNN models performs favorably, the blind DnCNN-B model performs poorly in removing the non-AWGN real noise.  
(The `better generalization ability of non-blind model` over blind one for controlling the trade-off between noise removal and detail preservation)  
- For figure 9, Noise clinic and BM3D fail to remove those structured noises since the structured noises fit the nonlocal self-similarity prior adopted in Noise Clinic and BM3D. In contrast, FFDNet and DnCNN successfully remove such noise without losing underlying image textures.  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure17.png" style="width:100%">
  <figcaption>
  Fig.10 - Color image denoising results by different methods on real noisy images from left to right: noisy image, denoised image by Noise Clinic, denoised image by CBM3D, denoised image by CDnCNN-B, denoised image by FFDNet (top) &sigma; = 28, (bottom) &sigma; = 15.
  </figcaption>
</p>

- CDnCNN-B yields very pleasing results for noisy image with AWGN-like noise such as image 'frog' and is still unable to handle non-AWGN noise.  
- We can conclude that while the nonlocal self-similarity prior helps to remove random noise, it hinders the removal of structured noise.  
- In comparison, the prior implicitly learned by CNN is able to remove both random noise and structured noise.  


<p>
  <img src="/assets/images/blog/Image_Denoising/Figure18.png" style="width:100%">
  <figcaption>
  Fig.11 - An example of FFDNet on image 'glass' with spatially variant noise. (a) Noisy image, (b) Denoised image by Noise Clinic; (c) Denoised image with FFDNet with &sigma; = 10, (d) Denoised image by FFDNet with &sigma; = 25, (e) Denoised image by FFDNet with &sigma; = 35, (f) Denoised image by FFDNet with non-uniform noise level map. milk-foam and specular reflection regions: &sigma; = 10, background region: &sigma; = 45, other regions: &sigma; = 25.
  </figcaption>
</p>

- While FFDNet `with a small uniform input noise level` can recover the details of regions with low noise level, it fails to remove strong noise.  
- FFDNet `with a large uniform input noise level` can remove strong noise but it will also smooth out the details in the region with low noise level.  
- The denoising result `with a proper non-uniform noise level map` not only preserves image details but also removes the strong noise.  


#### F. Running Time  

<p>
  <img src="/assets/images/blog/Image_Denoising/Table6.png" style="width:70%">
  <figcaption>
  Table.3 - Running time (in seconds) of different methods for denoising images with size 256 x 256, 512 x 512 and 1,024 x 1,024.
  </figcaption>
</p>

- Performed in Matlab (R2015b) with a six-core Intel(R) core(TM) i7-5820K CPU 3.3GHz, 32GB of RAM and Nvidia Titan X pascal GPU.  
- BM3D spends much more time on denoising color images than grayscale images.  
(compared to gray-BM3D, CBM3D needs extra time to denoise the chrominance components after luminance-chrominance color transformation)
- While DnCNN can benefit from GPU computation for fast implementation, it has comparable CPU time to BM3D.  
- FFDNet spends almost the same time for processing grayscale and color images.  


*Taking denoising performance and flexibility into consideration, FFDNet is very competitive for practical applications*.  

<br>
