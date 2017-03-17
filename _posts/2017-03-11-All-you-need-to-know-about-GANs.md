---
layout: post
title: All you need to know about GANs
date: 2017-03-17T00:00:00.000Z
published: true
---

Have you ever wanted to know about Generative Adversarial Networks (GANs)? Maybe you just want to be up to day on the topic? Or maybe you simply want to see how these networks have been refined over these last years. Well, in these cases, this post might interest you! 

#### What this post is not about
First things first, this is what you __won't__ find in this post:

* A tutorial about GANs? TODO
* Complex technical explanations
* Code (not explicitly: there are links to code for those interested)
* An exhaustive research list (you can already find it [here][GANpapers])

#### What this post is about
* A summary of the important things about GANs
* A lot of links to other sites, posts and articles so you can decide where to focus on
* Recent progress and potential future applications

# Understanding GANs

If you are reading this, chances are that you have heard GANs are pretty promising. Is the hype justified? This is what Yann LeCun, director of Facebook AI, thinks:

> "Generative Adversarial Networks is the most interesting idea in the last ten years in machine learning."

I personally think that GANs have a huge potential but we still need have a lot to figure out until we can reach to reach that point.

![All aboard the GAN train](https://cdn.meme.am/instances/500x/48663315.jpg){:height="auto" width="45%"}

In any case, what are GANs? I'm going to describe them very briefly. In case you are not familiar about them and want to know more details, there are a lot of great sites with good explanations. As a personal recommendation, I like the ones from [Eric Jang][introGAN1] and [Brandon Amos][introGAN2].

So, GANs — originally proposed by Ian Goodfellow — have two networks, a generator and a discriminator. They are both trained at the same time and compete again each other in a minimax game. The generator is trained to fool the discriminator creating realistic images, and the discriminator is trained not to be fooled by the generator.

![GAN training overview]({{site.baseurl}}/_posts/GAN_training_overview.jpg)

At first, the generator generates images. It does that by sampling a vector noise Z from a simple distribution (e.g. normal), and then upsampling this vector up to an image. In the first iterations, these images will look very noisy. 
Then, the discriminator is given fake and real images, and learns to distinguish them. The generator then receives the "feedback" of the discriminator with backpropagation, becoming better at generating images. At the end, we want that the distribution of fake images is as close as possible to the distribution of real images. Or, in simple words, we want fake images to look as plausible as possible.

It is worth mentioning that due to the minimax optimization used in GANs, the training might be quite unstable. There are some [hacks][GANhacks], though, that you can use for a more robust training.



### Code

If you are interested in the implementation of GANs, here are a bunch of links to short and simple codes:

* Tensorflow
* Torch and Python
* Torch and Lua

This is not the state-of-the-art code, but it is easy enough to understand the idea. If you are looking for the best implementation to make your own stuff, take a look at [this later section][TODO].

[GANpapers]: https://github.com/zhangqianhui/AdversarialNetsPapers
[introGAN1]: http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
[introGAN2]: https://bamos.github.io/2016/08/09/deep-completion/#ml-heavy-generative-adversarial-net-gan-building-blocks
[GANhacks]: https://github.com/soumith/ganhacks#authors
