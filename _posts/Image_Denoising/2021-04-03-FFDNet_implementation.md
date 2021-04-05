---
title:  "FFDNet paper_Review_Implementation"
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

In this post, I wrote the contents only about the implementations (networks, dataset and results). If you want to look into details about FFDNet, you can see them in the 'FFDNet Theory' post.  


<br>

### 1. Proposed Fast and Flexible Discriminative CNN Denoiser  

#### A. Network Architecture  

<p>
  <img src="/assets/images/blog/Image_Denoising/Figure8.png" style="width:100%">
  <figcaption>
  Fig.1 - The architecture of the proposed FFDNet for image denoising. The input image is reshaped to four sub-images, which are then input to the CNN together with a noise level map. The final output is reconstructed by the four denoised sub-images
  </figcaption>
</p>

- The first layer is a `reversible downsampling operator` (downsampling factor 2).  
- They further concatenate a tunable `noise level map M` with the downsampled sub-images (size `W/2 x H/2 x (4C+1)`) as the input to CNN.  
- Zero-padding for each layer  
- After the last convolution layer, an `upscaling operation` (sub-pixel convolution layer) is applied.  
- Different from DnCNN, FFDNet does not predict the noise.  
- filter size: 3 x 3  
- grayscale image: 15 layers, 64 channels of feature maps  
- Color image: 12 layers, 96 channels of feature maps  

<br>

### 2. Experiments  


#### A. Dataset Generation and Network Training  

FFDNet model is trained on the noisy images *y<sub>i</sub>* = *x<sub>i</sub>* + *v<sub>i</sub>* `without quantization to 8-bit integer values`.  

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


#### C. Experiments on Real Noisy Images  

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


#### D. Running Time  

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
