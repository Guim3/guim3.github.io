---
layout: post
title: Fantastic GANs and where to find them II
date: 2017-11-19
published: false
---

Hello again! This is the follow-up blog post of the original [Fantastic GANs and where to find them][original_post]. If you haven't checked that article or you are completely new to GANs, consider giving it a quick read - there's a brief summary of the previous post [ahead](#refresher), though. It has been 8 months since the last post and GANs aren't exactly known for being a field with few publications. In fact, I don't think we are very far from having more types of GAN names than Pokémon. Even Andrej Karpathy himself finds it difficult to keep up to date:

<blockquote class="twitter-tweet tw-align-center" data-lang="en"><p lang="en" dir="ltr">GANs seem to improve on timescales of weeks; getting harder to keep track of. Another impressive paper and I just barely skimmed the other 3</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/849135057788850177">4th of April 2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

So, having this in mind. Let's see what relevant advances have happened in these last months.

#### What this post is not about
This is what you __won't__ find in this post:

* Complex technical explanations.
* Code (links to code for those interested, though).
* An exhaustive research list (you can already find one [here][GANpapers]).

#### What this post is about
* A summary of relevant topics about GANs, starting where I left it on the [previous post](#original_post).
* A lot of links to other sites, posts and articles so you can decide where to focus on.

#### Index
1. [Refresher](#refresher)
2. [GANs: the evolution (part II)](#gans-evolution-II)
	1. [Improved WGANs](#impWGANs)
	2. [BEGANs](#BEGANs)
	3. [ProGANs](#ProGANs)
	4. [Honorable mention: CycleGANs](#honorable-mention)
3. [Other useful resources](#useful-resources)
4. [Closing](#closing)

## <a name="refresher"></a> Refresher

Let's get a brief refresher from the last post.

* **What are GANs**: two neural networks competing (and learning) against each other. Popular uses for GANs are generating realistic fake images, but they can also be used for unsupervised learning (e.g. learning features from data without labels).

![You don't need to design a loss function if a discriminator can design one for you]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/roll_safe_GANs.jpg){:height="auto" width="60%" .center-image}
GANs in a nutshell.
{: .img-caption}

* **Relevant models from previous post:**
	1. _Generative Adversarial Networks_: the original, vanilla, GANs.
	2. _Deep Convolutional GANs (DCGANs)_: first major improvement on the GAN architecture in terms of training stability and quality of the samples.
	3. _Improved DCGANs_: another improvement over the previous baseline, DCGANs. It allows generating higher-resolution images.
	4. _Conditional GANs (cGANs)_: GANs that use label information to enhance the quality of the images and control how these images will look.
	5. _InfoGANs_: this type is able to encode meaningful image features in a completely unsupervised way. For example, on the digit dataset MNIST, they encode the rotation of the digit.
	6. _Wasserstein GANs (WGANs)_: redesign of the original loss function, which correlates with image quality. This also improves training stability and makes WGANs less reliant on the network architecture.

## <a name="gans-evolution-II"></a> GANs: the evolution (part II)

Here I'm going to describe in chronological order the most relevant GAN articles that have been published lately. 

### <a name="impWGANs"></a> Improved WGANs (WGAN-GP)
<div class="date">March 2017</div>

**TL;DR:** take Wasserstein GANs and remove weight clipping - which is the cause of some undesirable behaviours - for gradient penalty. This results in faster convergence, higher quality samples and a more stable training. 

[[Article]][impWGAN_paper] [[Code]][impWGAN_code]

**The problem.** WGANs sometimes generate poor quality samples or fail to converge in some settings. This is mainly caused by the weight clipping (clamping all weights into a range [min, max]) performed in WGANs as a measure to satisfy the Lipschitz constraint. If you don't know about this constraint, just keep in mind that it's a requirement for WGANs to work properly. Why is weight clipping a problem? Because it biases the WGAN to use much simpler functions. This means that the WGAN might not be able to model complex data with simple approximations (see image below). Additionally, weight clipping makes vanishing or exploding gradients prone to happen.

![WGAN-GP 8 Gaussians toy example]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_8_Gaussians.jpg){:height="auto" width="380px" .center-image}
Here you can see how a WGAN fails to model 8 Gaussians (left) because it uses simple functions. On the other hand, a WGAN-GP correctly models them using more complex functions (right). 
{: .img-caption}

**Gradient penalty.** So how do we get rid of weight clipping? The authors of the WGAN-GP (where GP stands for gradient penalty) propose enforcing the Lipschitz constraint using another method which they call gradient penalty. Basically, GP consists of restricting some gradients to have a norm of 1. This is why it's call gradient penalty, as it penalizes gradients which norms deviate from 1.

**Advantages.** As a result, WGANs trained using GP rather than weight clipping have faster convergence. Additionally, the training is much more stable to an extent where hyperparameter tuning is no longer required and the architecture used is not as critical. These WGAN-GP also generate high-quality samples, but it is difficult to tell by how much. On proven and tested architectures, the quality of these samples are very similar to the baseline WGAN:

![WGAN-GP baseline comparison]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_comparison_DCGAN.jpg){:height="auto" width="500px" .center-image}
{: .img-caption}

Where WGAN-GP is clearly superior is on generating high-quality samples on architectures where other GANs are prone to fail. For example, to the authors' knowledge, it has been the first time where a GAN setting has worked on residual network architectures:

![WGAN-GP other architectures comparison]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/WGAN-GP_comparison_other.jpg){:height="auto" width="500px" .center-image}
{: .img-caption}

There are a lot of other interesting details that I had not mentioned, as it'd go far beyond the scope of this post. For those that want to know more (e.g. why the gradient penalty is applied just to "some" gradients or how to a apply this model to text), I recommend taking a look at the [article][impWGAN_paper].

#### You might want to use WGANs-GP if
you want an improved version of the WGAN which

* converges faster.
* works on a wide variety of architectures and datasets.
* doesn't require as much hyperparameter tuning as other GANs.


### <a name="BEGANs"></a> Boundary Equilibrium GANs (BEGANs)
<div class="date">March 2017</div>

**TL;DR:** GANs using an auto-encoder as the discriminator. They can be successfully trained with simple architectures. They incorporate a dynamic term that balances both discriminator and generator during training. 

[[Article]][BEGAN_paper]

_Fun fact: BEGANs were published on the very same day as the WGAN-GP paper._

**Idea.** What sets BEGANSs apart from other GANs is that they use an auto-encoder architecture for the discriminator (similarly to [EBGANs][EBGANs]) and a special loss adapted for this scenario. What is the reason behind this choice? Are auto-encoders not the devil as they force us to have a pixel reconstruction loss that makes [blurry generated samples][VAEs_blurry]? To answer these questions we need to consider these two points:

1. Why reconstruction loss? The explanation from the authors is that we can rely on the assumption that matching the reconstruction loss distribution will end up matching the sample distributions.

2. Which leads us to: how? An important remark is that the reconstruction loss from the auto-encoder/discriminator (i.e. given this input image, give me the best reconstruction) is not the final loss that BEGANs are minimizing. This reconstruction loss is just a step to calculate the final loss. And the final loss is calculated using the Wasserstein distance (yes, it's everywhere now) between the reconstruction loss on real and generated data.

This might be a lot of information at once, but I'm sure that, once we see how this loss function is applied to the generator and discriminator, it'll be much clearer:

* The generator focuses on generating images that the discriminator will be able to reconstruct well.
* The discriminator tries to reconstruct real images as good as possible while reconstructing generated images with the maximum error.

**Diversity factor.** Another interesting contribution is what they call the diversity factor. This factor controls how much you want the discriminator to focus on getting a perfect reconstruction on real images (quality) vs distinguish real images from generated (diversity). Then, they go one step further and use this diversity factor to maintain a balance between the generator and discriminator during training. Similarly to WGANs, they use this equilibrium between both networks as a measure of convergence that correlates with image quality. However, unlike WGANs (and WGANs-GP), they use Wasserstein distance in such a way that the Lipschitz constrain is not required.

**Results.** BEGANs do not need any fancy architecture to train properly; as mentioned in the paper: "no batch normalization, no dropout, no transpose convolutions and no exponential growth for convolution filters". The quality of the generated samples (128x128) is impressive*:

![BEGAN face samples]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/BEGAN_samples.jpg){:height="auto" width="400px" .center-image}
BEGANs realistic generated images.
{: .img-caption}

*However, there's an important detail to be considered in this paper. They are using an unpublished dataset which is almost twice the size of the widely used [CelebA][celeba] dataset. Then, for a more realistic qualitative comparison, I invite you to check [any public implementation][BEGAN-tf] using CelebA and see the generated samples.

As a final note, if you want to know more about BEGANs, I recommend reading this [blog post][BEGAN_blogpost], which goes much more into detail.

#### You might want to use BEGANs...
... for the same reasons you would use WGANs-GP. They both offer very similar results (stable training, simple architecture, loss function correlated to image quality), they mainly differ in their approach. Due to the hard nature of evaluating generative models, it's difficult to say which is better. As Theis et al. says in [their paper][Theis], you should choose a evaluation method or another depending on the application. In this case, WGAN-GP has a better Inception score and yet BEGANs generate very high-quality samples. Both are innovative and promising.

### <a name="ProGANs"></a> Progressive growing of GANs (ProGANs)
<div class="date">October 2017</div>

**TL;DR:** Progressively add new high-resolution layers during training that generates incredibly realistic images. Other improvements and a new evaluation method are also proposed. The quality of the generated images is astonishing.

[[Article]][ProGANs_article] [[Code]][ProGANs_code]

Generating high-resolution images is a big challenge. The larger the image, the easier is for the network to fail because the details are more subtle and complex to model. To give a little bit of context, before this article, realistic generated images were around 256x256. Progressive GANs (ProGANs) take this to a whole new level by successfully generating completely realistic 1024x1024 images. Let's see how.

**Idea.** ProGANs, which are built upon [WGANs-GP](#impWGANs), introduce a smart way to progressively add new layers on training time. Each one of these layers upsamples the images to a higher resolution for both the discriminator and generator. Let's go step by step:

1. Start with the generator and discriminator training with low-resolution images.
2. At some point (e.g. when they start to converge) increase the resolution. This is done very elegantly with a "transition period" / smoothing:

![ProGANs smoothing]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/proGANs_smoothing.jpg){:height="auto" width="600px" .center-image}
{: .img-caption}

<p style="margin-left: 23px">Instead of just adding a new layer directly, it's added on small linear steps controlled by α.</p>
<p style="margin-left: 23px">Let's see what happens in the generator. At the beginning, when α = 0, nothing changes. All the contribution of the output is from the previous low-resolution layer (16x16). Then, as α is increased, the new layer (32x32) will start getting its weights adjusted through backpropagation. By the end, α will be equal to 1, meaning that we can totally drop the "shortcut" used to skip the 32x32 layer. The same happens to the discriminator, but the other way around: instead of making the image larger, we make it smaller.</p>

<ol start="3">
  <li>Once the transition is done, keep training the generator and discriminator. Go to step 2 if the resolution of currently generated images is not the target resolution</li>
</ol>

**But, wait a moment...**
isn't this upsampling and concatenation of new high-resolution images something already done in [StackGANs][StackGANs] (and the new [StackGANs++][StackGAN++])? Well, yes and no. First of all, StackGANs are text-to-image conditional GANs that use text descriptions as an additional input while ProGANs don't use any kind of conditional information. But, more interestingly, despite both StackGANs and ProGANs using concatenation of higher resolution images, StackGANs require as many independent pairs of GANs - which need to be trained separately - per upsampling. Do you want to upsample 3 times? Train 3 GANs. 
On the other hand, in ProGANs only a single GAN is trained. During this training, more upsampling layers are *progressively* added to upsample the images. So, the cost of upsampling 3 times is just adding more layers on training time, as opposed to training from scratch 3 new GANs. In summary, ProGANs use a similar idea from StackGANs and they manage to pull it off elegantly, with better results and without extra conditional information.

**Results.** As a result of this progressive training, generated images in ProGANs have higher quality and training time is reduced by 5.4x (1024x1024 images). The reasoning behind this is that a ProGAN doesn't need to learn all large-scale and small-scale representations at once. In a ProGAN, first the small-scale are learnt (i.e. low-resolution layers converge) and then the model is free to focus on refining purely the large-scale structures (i.e. new high-resolution layers converge).

<iframe width="560" height="315" src="https://www.youtube.com/embed/XOxxPcy5Gr4" frameborder="0" allowfullscreen></iframe>{: .center-image }
The resulting generated images are clearly superior to any other GAN seen before.
{: .img-caption}

**Other improvements**. Additionally, the paper proposes new design decisions to further improve the performance of the model. I'll briefly describe them:

* Minibatch standard deviation: encourages each minibatch to have similar statistics using the standard deviation over all features of the minibatch. This is then summarized as a single value in a new layer that is inserted towards the end of the network.

* Equalized learning rate: makes sure that the learning speed is the same for all weights by dividing each weight by a constant continuously during training.

* Pixelwise normalization: on the generator, each feature vector is normalized after each convolutional layer (exact formula in the paper). This is done to prevent the magnitudes of the gradients of the generator and discriminator from escalating.

**CelebA-HQ**. As a side note, it is worth mentioning that the authors enhanced and prepared the original CelebA for high-resolution training: CelebA-HQ. In a nutshell, they remove artifacts, apply a Gaussian filtering to produce a depth-of-field effect, and detect landmarks on the face to finally get a 1024x1024 crop. After this process, they only keep the best 30k images out of 202k.

**Evaluation**. Finally, a new evaluation method is introduced:
* The idea behind it is that the local image structure of generated images should match the structure of the training images. 
* How do we measure local structure? With a Laplacian pyramid, where you get different levels of spatial frequency bands that can be used as descriptors. 
* Then, we extract descriptors from the generated and real images, normalize them, and check how close they are using the famous Wasserstein distance. The lower the distance, the better.

#### You might want to use ProGANs...
* If you want state-of-the-art results. But consider that...
* ... you will need *a lot* of time to train the model: "We trained the network on a single NVIDIA Tesla P100 GPU for 20 days".
* If you want to start questioning your own reality. The next iterations on GANs will might create more realistic samples than real life.

#### <a name="honorable-mention"></a> Honorable mention: Cycle GANs

[[Article]][CycleGANs_article] [[Code]][CycleGANs_code]

Cycle GANs are, at the moment of writing these words, the most advanced image-to-image translation using GANs. Tired that your horse is not a zebra? Or maybe that Instagram photo needs more winter? Cycle GANs are the answer. 

![ProGANs smoothing]({{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/CycleGANs_results.jpg){:height="auto" width="650px" .center-image}
{: .img-caption}

These GANs don't require paired datasets to learn to translate between domains, which is good because this kind of data is very difficult to obtain. However, Cycle GANs still need to be trained with data from two different domains X and Y (e.g. X: horses, Y: zebras). In order to constrain the translation from one domain to another, they use what they call a "cycle consistent loss". This basically means that if you translate a horse A into a zebra A, transforming the zebra A back to a horse should give you the original horse A as a result.

This mapping from one domain to another is different from the also popular [neural style transfer][neural_style_transfer]. The latter combines the content of one image with the style of another, whilst Cycle GANs learn a high level feature mapping from one domain to another. As a consequence, Cycle GANs are more general and can also be used for all sorts of mappings such as converting a sketch of an object into a real object.

![]({{site.baseurl}}/files/blog/common/separator1.png){: .center-image}

Let's recap. We have had two major improvements, WGANs-GP and BEGANs. Despite following different research directions, they both offer similar advantages. Then, we have ProGANs (based on WGANs-GP), which unlock the path to generate realistic high-resolution images. Meanwhile, CycleGANs reminds us about the power of GANs to extract meaningful information from a dataset and how this information can be transferred to another unrelated data distribution. 

## <a name="useful-resources"></a> Other useful resources

Here are a bunch of links to other interesting posts:

* [GAN playground][GAN_playground]: this is the most straightforward way to play around GANs. Simply click the link, set up some hyperparameters and train a GAN in your browser.
* [Every paper and code][GANs_everything]: here's a link to all GAN related papers sorted by the number of citations. It also includes courses and Github repos. Very recommended, but the last update was on July 2017. 
* [GANs timeline][GANs_timeline]: similar to the previous link, but this time every paper is ordered according to publishing date.
* [GANs comparison][GANs_no_cherry]: in this link, different versions of GANs are tested without cherry picking. This is a important remark, as generated images shown in publications might not be really representative of the overall performance of the model.
* [Some theory behind GANs][GAN_theories]: in a similar way to this post, this link contains some nice explanations of the theory (especially the loss function) of the main GAN models.
* [High-resolution generated images][high_res_GANs]: this is more of a curiosity, but here you can actually see how 4k x 4k generated images actually look like.
* [Waifus generator][waifu_generator]: you'll never feel alone ever again ( ͡° ͜ʖ ͡°)

<br>
![]({{site.baseurl}}/files/blog/common/separator2.png){:height="auto" width="250px" .center-image}
<br>

<a name="closing"></a> 
Hope this post has been useful and thanks for reading!! I want to also say thanks to Blair Young for helping me improving this post with his feedback. If you think there's something wrong, inaccurate or want to make any suggestion, please let me know in the comment section below or in [this reddit thread][reddit].

As a side note, I'm currently living in London. If you are in town and a GAN nerd like me and want to talk about all types of GANs, complain about the evaluation of generative models or want my opinion about your groundbreaking state-of-the-art meme generator, drop me an email! I'm always happy to grab a drink and share experiences.

Oh, and I have just created my new [twitter account][twitter]. I'll be sharing my new blog posts there (and some dank convolutional memes). 

[original_post]: http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them
[GANpapers]: https://github.com/zhangqianhui/AdversarialNetsPapers
[GANs_everything]: https://github.com/GKalliatakis/Delving-deep-into-GANs
[GAN_theories]: https://github.com/YadiraF/GAN_Theories
[GANs_no_cherry]: https://github.com/khanrc/tf.gans-comparison
[high_res_GANs]: http://mtyka.github.io/machine/learning/2017/06/06/highres-gan-faces.html
[impWGAN_paper]: https://arxiv.org/abs/1704.00028
[impWGAN_code]: https://github.com/igul222/improved_wgan_training
[EBGANs]: https://arxiv.org/abs/1609.03126
[BEGAN_paper]: https://arxiv.org/abs/1703.10717
[VAEs_blurry]: {{site.baseurl}}/files/blog/Fantastic-GANs-and-where-to-find-them-II/VAEs_blurred_samples.jpg
[BEGAN_blogpost]: https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/
[celeba]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
[BEGAN-tf]: https://github.com/carpedm20/BEGAN-tensorflow
[CycleGANs_article]: https://arxiv.org/pdf/1703.10593.pdf
[CycleGANs_code]: https://github.com/junyanz/CycleGAN
[Theis]: https://arxiv.org/abs/1511.01844
[StackGAN++]: https://arxiv.org/abs/1710.10916
[StackGAN++_code]: https://github.com/hanzhanggit/StackGAN-v2/
[GAN_playground]: https://reiinakano.github.io/gan-playground/
[ProGANs_article]: https://arxiv.org/abs/1710.10196
[ProGANs_code]: https://github.com/tkarras/progressive_growing_of_gans
[StackGANs]: http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them#StackGANs
[waifu_generator]: http://make.girls.moe/#/
[GANs_timeline]: https://github.com/dongb5/GAN-Timeline
[neural_style_transfer]: https://github.com/jcjohnson/neural-style
[twitter]: https://twitter.com/GuimPML
[reddit]: TODO
