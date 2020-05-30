---
layout: post
title:  Visualizing Model Comparison
date:   2020-05-27 23:00
categories: [blogpost]
tags: representational_similarity_analysis multi_dimensional_scaling neural_networks
comments: true
mathjax: true
excerpt: In this post, I explain the representational analysis technique and how we can use it in combination with a multi dimensional scaling algorithm to visualize the similarity between multiple models, or different instances of the same model in 2d.
image:
  feature: repsim_images/repsim_alg.png
---

When training deep neural networks, we often strive to compare the models
and understand the consequences of different design choices beyond the final performance on the models on a given task.
For example, it is can be very insightful to visualize the internal representations of the models in 2D or 3D
using dimensionality reduction techniques such as [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) or [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).

But when we want to see differences in the representational spaces of different models, it can be tricky to just compare the plots by eyes. To quantify the similarity between the representational spaces of two models, we can apply
**representational similarity analysis** (RSA) {% cite repsimlaakso2000 %}, which in simple terms is to compute the similarity of similarities between two spaces. This is in particular useful when these models do not have the same architecture and their parameter space is not directly comparable. Thus, if we have a set of models and we want to know how they are all compared to each other in terms of representational similarity, we can compute the representational similarity between each model pair. This will give us the similarity matrix of the models and we can
visualize this in 2d or 3d using a [multi dimensional scaling algorithm](https://en.wikipedia.org/wiki/Multidimensional_scaling). I have depicted this process in the figure below.


{% include image.html
            img="img/repsim_images/repsim_alg.png"
            title="Representational similarity"
            caption="Computing representational similarity of multiple models"
            width="800px" %}

Assume we have $m$ models, and we want to visualize them in 2d based on the similarity of their penultimate layers.
* First, we feed a sample set of size $n$ from the validation/test set (e.g. 1000 examples) to the forward pass of each model and obtain the representation from a particular layer, e.g., penultimate layer, of the models.
* Next, for each model, we calculate the similarity of the representations of all pairs from the sample set using some similarity metric, e.g., dot product. This leads to a matrix of size $n\times n$ (part 1 in the figure above).
* We use the samples similarity matrix associated with each model to compute the similarity between all pairs of models. Thus, we compute the dot product (we can use any other measure of similarity as well) of the corresponding rows of these two matrices after normalization, and average all the similarity of all rows, which leads to a single scalar (part 2 and 3 in the figure above).
* Given all possible pairs of models, we then have a model similarity matrix of size $m\times m$ and we apply a [multi dimensional scaling algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) to embed all the models in a 2D space based on their similarities.

In the example below we see the similarity between the penultimate layers of the models trained on the MNIST dataset.
{% include image.html
            img="img/repsim_images/mnist_repsim.png"
            title="Representational similarity"
            caption="Representational similarity of different instances of MLPs and CNNs trained on MNIST"
            width="400px" %}

We can use this technique not only to compare different models but also to compare different components of the same model, e.g., different layers, or to see how the representations of a model change/evolve during training. For example, in figure below, we can see the training trajectories of a CNN and an MLP on the MNIST dataset. To obtain these training trajectories, we compute the pairwise similarity between the representations obtained from the models at different stages of the training process.

{% include image.html
            img="img/repsim_images/training_path.png"
            title="Representational similarity"
            caption="Visualizing training paths of models using RSA"
            width="800px" %}
