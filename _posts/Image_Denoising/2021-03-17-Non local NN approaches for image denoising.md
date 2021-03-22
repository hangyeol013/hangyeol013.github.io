---
title:  "Non local neural network approaches for image denoising"
search: true
categories:
  - Deep learning
  - Image denoising
classes: wide
summary: This post is a summary of image denoising review paper.
last_modified_at: 2021-03-17T08:06:00-05:00
---


This post is the seventh part (RNN) of summary of a survey paper
[A state-of-the Art survey on Deep learning theory and architecture](https://www.mdpi.com/2079-9292/8/3/292).  

Super resolution is the tasking of mapping a low resolution image to a high resolution image whereas image denoising is the task of learning a clean image from a noisy input.  


### 1. Introduction  

Image denoising is one of the most standard `inverse problem` in image processing. The first to propose a non local approach to this problem called Non-Local Means (NLM). The main idea is to exploit the `self similarity` properties of natural images. It is easy to observe that patterns such as edges or textures, are repeated with light variations along the image. Such patches which may not be close to each other in the image may contain very similar information. NLM computes for each pixel the weighted average of the patches of the image, where the weights depend on the similarity of the patches. Other methods such as NL-Bayes or BM3D also exploit this property.  

Those methods work very well on images with strong self similarity, but perform poorly on random micro-textures where self similarity is low.  

Denoising CNNs such as DnCNN or FFDNET haver out-performed those non local methods in terms of PSNR. We still observe an `over-smooting of textures` for those methods and poor performances on images with strong self-similarity.  

Among many reasons, one is the local nature of convolutions. Even if the author claim to have large receptive fields, the importance of a pixel *j* for the computation of a pixel *i* decreases significantly as *j* gets away from *i*. This is one of the major flaws of CNNs for image denoising.  

There has been an emergence of denoising methods which use CNNs and non local approaches simultaneously to be able to denoise homogeneous zones, textures, and images with strong self-similarity. The paper report and explain here some of those methods which can be split in three categories.  
1. Plug-and-play methods which are agnostic of the chosen denoising CNN architecture.  
2. Non local inference CNNs which unroll classic non-local denoising algorithm into a CNN with block-matching methods.  
3. The architecture including attention mechanisms, with non-local layers and kNN searches for example.  


<br>


### 2. Plug-and-play non local approaches  

They define plug-and-play methods as denoising algorithms which can include any type of denoising methods. In that case, we can use any existing CNNs in those approaches. The `versatility` and the `ease of implementation` are the major benefits of those PNP methods.  

**Block Matching CNN**
This method can be split in two: a block matching algorithm and a denoising algorithm. The block matching algorithm selects for each pixel the k-NN patches in a defined search window, using the euclidean distance. The k-NN are not computed on the noisy image but on a pilot image, obtained thanks to already existing denoising methods (e.g., BM3D, DnCNN) Computing the block-matching on the noisy image leads to very poor results.  

Once the k-NN are found, they concatenate both the noisy and the pilot patches in a 3D tensor of size (*2k, N<sub>patch</sub>, N<sub>patch</sub>*) that is further passed through a regular and trainable CNN that outputs a single patch. The final image from which the loss is computed, is obtained using a classic patch aggregation method. This method shows outperforming DnCNN but the computation time is much larger since the NN search is time consuming.  

Similarly, a non local PNP method for video denoising was presented. This method can also be divided in pixel matching and CNN filtering. The difference is that the search window is now three-dimensional. Instead of stacking patches together, they stack the noisy frame t with its N nearest neighbor feature maps. Therefore the N most similar pixels to pixel (*i,t*) are stacked along the z axis at the same position *i*.  

Those N+1 feature maps are then passed through a CNN. First, 4 layers of 32 1x1 convolution process the non local features maps. The output is passed through a DnCNN-like network.

The interest of using multiple frames for video denoising, is that instead of using a pilot denoised image, we can just increase the patch-size when searching for the k-NNs, since videos have even more self-similarity than images, due to their temporal redundancy.  
