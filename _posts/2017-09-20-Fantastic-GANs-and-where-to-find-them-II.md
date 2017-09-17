---
layout: post
title: Fantastic GANs and where to find them II
date: 2017-09-20
published: false
---

Hello again! This is the follow-up blog post of the original [Fantastic GANs and where to find them](#original_post). If you haven't checked that article or you are completely new to GANs, it might be helpful if you give it a quick read. It has been 6 months since the last post and GANs aren't exactly known for being a field with few publications. In fact, I don't think we are very far from having more types of GAN names than Pokémons. Even Andrej Karpathy himself finds it difficult to being up to date:

<blockquote class="twitter-tweet" data-lang="es"><p lang="en" dir="ltr">GANs seem to improve on timescales of weeks; getting harder to keep track of. Another impressive paper and I just barely skimmed the other 3</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/849135057788850177">4th of April 2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

So, having this in mind, it's always a good idea to take a look at the literature and get a summary of the most promising advances so far. Let's do this.

Don't worry. Get yourself comfortable. Let's see what relevant advances have happened in these lasts 6 months so far.

#### What this post is not about
This is what you __won't__ find in this post:

* Complex technical explanations
* Code (links to code for those interested, though)
* An exhaustive research list (you can already find one [here][GANpapers])

#### What this post is about
* A summary of relevant topics about GANs, starting where I left it on the [previous post](#original_post).
* A lot of links to other sites, posts and articles so you can decide where to focus on.

#### Index
1. [Refresher](#refresher)
2. [GANs: the evolution (part II)](#gans-evolution)
	1. [DCGANs](#dcgans)
	2. [Improved DCGANs](#improved-dcgans)
	3. [Conditional GANs](#cGANs)
	4. [InfoGANs](#infoGANs)
	5. [Wasserstein GANs](#wassGANs)
3. [Other useful resources](#useful-resources)
3. [Closing](#closing)

## <a name="refresher"></a> Refresher

Let's get a brief refresher from the last post.

* **What are GANs**: two neural networks competing (and learning) against each other. Popular uses for GANs are generating fake images, but they can also be used for unsupervised learning (e.g. learn features from your data without labels).

![You don't need to design a loss function if a discriminator can design one for you]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/roll_safe_GANs.jpg){: .center-image }
GAN training overview.
{: .img-caption}

* **Relevant models**
	1. Generative Adversarial Networks: the original, vanilla, GANs.
	2. Deep Convolutional GANs (DCGANs): first major improvement on the GAN architecture in terms of training stability and quality of the samples.
	3. Improved DCGANs: another improvement over the previous baseline, DCGANs. It allows to generate higher-resolution images.
	4. Conditional GANs (cGANs): GANs that use label information to enhance the quality of the images and control how these images will look.
	5. Wassertein GANs (WGANs): redesign of the original loss function, which correlates with image quality. This also improves training stability and makes WGANs less reliant on the architecture.

## <a name="gans-evolution"></a> GANs: the evolution (part II)


### <a name="dcgans"></a>BEGANs?
**TL;DR:**  

[[Article]][DCGAN_art]

BEGAN

EBGAN?

InfoGAN?

CoGANs?

Boundary Seeking GAN?

Least Squares GAN?

DiscoGANs?

CasualGAN?  
https://arxiv.org/abs/1709.02023 

//Mira aquest link per explicació de Boundary Seeking, Least Squares GANs i CoGANs
https://wiseodd.github.io/techblog/

## <a name="useful-resources"></a> Other useful resources

Here are a bunch of links to other interesting posts:

* [GANs comparison](#GANS_no_cherry): in this link, different versions of GANs are tested without cherry picking. This is a important remark, as generated images shown in publications couldn't be really representative of the overall performance of the model.
* [Some theory behind GANs](#GAN_theories): in a similar way to this post, this link contains some nice explanations of the theory (specially the loss function) of the main GAN models.
* [High resolution generated images](high_res_GANs): this is more of a curiosity, but here you can actually see how 4kx4k generated images actually look like (usually, they aren't larger than 256x256).

[original_post]: http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them
[GANpapers]: https://github.com/zhangqianhui/AdversarialNetsPapers
[GAN_theories]: https://github.com/YadiraF/GAN_Theories
[GANs_no_cherry]: https://github.com/khanrc/tf.gans-comparison
[high_res_GANs]: http://mtyka.github.io/machine/learning/2017/06/06/highres-gan-faces.html