---
layout: post
title: Fantastic GANs and where to find them II
date: 2017-09-20
published: false
---

Hello again! This is the follow-up blog post of the original [Fantastic GANs and where to find them][original_post]. If you haven't checked that article or you are completely new to GANs, it might be helpful if you give it a quick read. It has been 6 months since the last post and GANs aren't exactly known for being a field with few publications. In fact, I don't think we are very far from having more types of GAN names than Pokémons. Even Andrej Karpathy himself finds it difficult to be up to date:

<blockquote class="twitter-tweet tw-align-center" data-lang="en"><p lang="en" dir="ltr">GANs seem to improve on timescales of weeks; getting harder to keep track of. Another impressive paper and I just barely skimmed the other 3</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/849135057788850177">4th of April 2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

So, having this in mind, it's always a good idea to take a look at the literature and get a summary of the most promising advances so far. Let's do this.

Don't worry. Get yourself comfortable. Let's see what relevant advances have happened in these last 6 months so far.

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
2. [GANs: the evolution (part II)](#gans-evolution-II)
	1. [Improved WGANs](#impWGANs)
	2. [BEGANs](#BEGANs)
3. [Other useful resources](#useful-resources)
4. [Closing](#closing)

## <a name="refresher"></a> Refresher

Let's get a brief refresher from the last post.

* **What are GANs**: two neural networks competing (and learning) against each other. Popular uses for GANs are generating fake images, but they can also be used for unsupervised learning (e.g. learn features from your data without labels).

![You don't need to design a loss function if a discriminator can design one for you]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/roll_safe_GANs.jpg){:height="auto" width="60%" .center-image}
GANs in a nutshell.
{: .img-caption}

* **Relevant models from previous post**
	1. __Generative Adversarial Networks__: the original, vanilla, GANs.
	2. __Deep Convolutional GANs (DCGANs)__: first major improvement on the GAN architecture in terms of training stability and quality of the samples.
	3. __Improved DCGANs__: another improvement over the previous baseline, DCGANs. It allows generating higher-resolution images.
	4. __Conditional GANs (cGANs)__: GANs that use label information to enhance the quality of the images and control how these images will look.
	5. __InfoGANs__: this type is able to encode meaningful image features in a completely unsupervised way. For example, on the digit dataset MNIST, they encode the rotation of the digit.
	6. __Wassertein GANs (WGANs)__: redesign of the original loss function, which correlates with image quality. This also improves training stability and makes WGANs less reliant on the network architecture.

## <a name="gans-evolution-II"></a> GANs: the evolution (part II)


### <a name="impWGANs"></a> Improved WGANs (WGAN-GP)
**TL;DR:** take Wassertein GANs and remove weight clipping - which is the cause of some undesirable behaviours - for gradient penalty. This results in faster convergence, higher quality samples and a more stable training. 

[[Article]][impWGAN_paper] 

WGANs sometimes generate poor quality samples or fail to converge in some settings. This is mainly caused by the weight clipping performed in WGANs as a measure to satisfy the Lipschitz constraint. If you don't know about this constraint, just keep in mind that it's a requirement for WGANs to work properly. Why is weight clipping a problem? Because it biases the WGAN to use much simpler functions. This means that the WGAN might not be able to model a complex data with simple approximations (see image below). Additionally, weight clipping makes vanishing or exploding gradients prone to happen.

![WGAN-GP 8 Gaussians toy example]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_8_Gaussians.jpg){:height="auto" width="380px" .center-image}
Here you can see how a WGAN fails to model 8 Gaussians because it uses simple functions. On the other hand, a WGAN-GP correctly models them using more complex functions. 
{: .img-caption}

So how do we get rid of weight clipping? The authors of the WGAN-GP (where GP stands for gradient penalty) propose enforcing the Lipschitz constraint using another method which they call gradient penalty. Basically, GP consists of restricting some gradients to have a norm of 1. This is what they call gradient penalty, as it penalizes gradients which norms deviate from 1.

As a result, WGANs trained using GP rather than weight clipping have faster convergence. Additionally, the training is much more stable to an extent where hyperparameter tuning is no longer required and the architecture used is not as critical. These WGAN-GP also generate high quality samples, but it is difficult to tell by how much. On proven and tested architectures, the quality of these samples are very similar to the baseline WGAN:

![WGAN-GP baseline comparison]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_comparison_DCGAN.jpg){:height="auto" width="500px" .center-image}
{: .img-caption}

Where WGAN-GP is clearly superior is on generating high quality samples on architectures where other GANs clearly fail. For example, to the authors knowledge, it has been the first time where a GAN setting has worked on Residual Networks:

![WGAN-GP other architectures comparison]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_comparison_other.jpg){:height="auto" width="500px" .center-image}
{: .img-caption}


	Result: faster convergence, higher quality samples, no hyperparameter tuning required.

	Contributions: prove weight clipping leads to pathological behaviour. WGAN with gradient penalty. Stable training with difficult architecture settings (e.g. very deep residual networks work for first time in GAN setting).

There are a lot of other interesting details that I had not mentioned, as it'd would go far beyond the scope of this post. For those that want to know more (e.g. why the gradient penalty is applied just to "some" gradients or how to a apply this improved WGAN to text), I recommend taking a look at the [article][impWGAN_paper].

#### You might want to use WGANs-GP if
you want an improved version of the WGAN (that redundancy) which

* converges faster.
* is not as dependant on the architecture used.
* doesn't require as much hyperparameter tuning as other GANs.
* gets to generate high quality samples on scenarios where other GANs fail.


BEGANs?
**TL;DR:**  

https://www.reddit.com/r/MachineLearning/comments/633jal/r170310717_began_boundary_equilibrium_generative/dfrktje/

[[Article]][DCGAN_art]

Improved WGANs

BEGAN

Check state of the art of conditional GANs (Reed?)

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
* [High resolution generated images](high_res_GANs): this is more of a curiosity, but here you can actually see how 4k x 4k generated images actually look like (usually, they aren't larger than 256 x 256).

[original_post]: http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them
[GANpapers]: https://github.com/zhangqianhui/AdversarialNetsPapers
[GAN_theories]: https://github.com/YadiraF/GAN_Theories
[GANs_no_cherry]: https://github.com/khanrc/tf.gans-comparison
[high_res_GANs]: http://mtyka.github.io/machine/learning/2017/06/06/highres-gan-faces.html
[impWGAN_paper]: https://arxiv.org/abs/1704.00028