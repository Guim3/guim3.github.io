---
layout: post
title: Fantastic GANs and where to find them III
date: 2019-02-15
published: false
---

[Text link][reference_link]
[Text to section][#tag_inside_section]



It has been a while since the last post! This is the third part of a series of blog posts where I talk about the research highlights about GANs. Are you interested in knowing the origins? Check the [first part][fantasticI]. Or maybe you want to see how GANs went from "that face looks sort of real" to "holy shit that could be my neighbour"? The [second part][fantasticI] is for you. In this blog post we are going to see how GANs manage to generate more realistic images than real life.

#### What this post is not about
This is what you __won't__ find in this post:

* Complex technical explanations. Intuition comes first.
* Code (links to code for those interested, though).
* An exhaustive research list (you can already find one [here][GANpapers]).

#### What this post is about
* A summary of relevant topics about GANs, starting where I left it on the [previous post][fantasticII] (ProGANs).
* A lot of links to other sites, posts and articles so you can decide where to focus on.

#### Index
1. [Refresher?](#refresher)
2. [GANs: the evolution (part III)](#gans-evolution-III)
	1. [sGANs](#sGANs)
	2. [sGANs](#sGANs)
	3. [](#sGANs)
3. [Honourable mentions](#honourable-mentions)
	1. [Are GANs created equal?](#equal-GANs)
	2. [GAN dissection](#GAN-dissection)
4. [Other useful resources](#useful-resources)
5. [Closing](#closing)

## <a name="refresher"></a> Refresher

Let's get a brief refresher from the last post. Or maybe not.

## <a name="gans-evolution-III"></a> GANs: the evolution (part III)

As always, here I describe in chronological order the GAN articles that have been a break-through in the field.

### <a name="tagReference"></a> Something GANs (sGANs)
<div class="date">March 201x</div>

**TL;DR:** blablabla

[[Article]][paper_link] [[Code]][paper_code]

#### You might want to use sGANs if

### <a name="tagReference"></a> Something GANs (sGANs)
<div class="date">March 201x</div>

**TL;DR:** blablabla

[[Article]][paper_link] [[Code]][paper_code]

#### You might want to use sGANs if

### <a name="tagReference"></a> Something GANs (sGANs)
<div class="date">March 201x</div>

**TL;DR:** blablabla

[[Article]][paper_link] [[Code]][paper_code]

#### You might want to use sGANs if

## <a name="useful-resources"></a> Other useful resources

Here are a bunch of links to other interesting posts:


## <a name="honourable-mentions"></a> Honourable mentions

### <a name="GAN-dissection"></a> GAN dissection: visualizing and understanding GANs
<div class="date">November 2018</div>

**TL;DR:** blablabla

[[Article]][paper_link] [[Code]][paper_code]

Let's forget about the paper for a moment. Let's try to solve the problem presented in this paper ourselves. The premise is the following: how can we open up GANs and see what is inside of them? How can we know which neurons are responsible of, say, creating a tree?

First off, we can start by trying to relate a specific part of the network (a subset of neurons) to a semantic concept (e.g. a tree) generated in the output image. In other words, our GAN is generating a tree, but where and how is this tree being generated? In order to find the answer, let's first find the three pieces of this puzzle: a set of neurons, a concept, and the relation between these two. We already have access to an arbritary set of neurons inside the generator, so that's done. 
What about the concept? How do we know if an image has a tree, and where? This is more tricky! We need to know what each pixel in the image represents. Is that pixel part of a tree, or is it just sky? For the sake of the example, let's assume we have a dataset containing all this pixel information. 
There's one last element missing now: relating neurons with concepts. That sounds too abstract, though. Now that we have a dataset, we can think of it as finding how a set of neurons affect the outcome of producing tree pixels.

**Relating neurons to generated pixels.** 
Key idea: generate feature map that matches generated image.

Let's recap:
*

**Dissection**


**Intervention**
Effect of turning a set of units on and off in the generated image.


<br>
![]({{site.baseurl}}/files/blog/common/separator2.png){:height="auto" width="250px" .center-image}
<br>

<a name="closing"></a> 


[fantasticI]: http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them
[fantasticII]: http://guimperarnau.com/blog/2017/11/Fantastic-GANs-and-where-to-find-them-II

