---
layout: post
title: Experiential, Distributional and Dependency-based Word Embeddings have Complementary Roles in Decoding Brain Activity
date:   2018-01-07 23:00
categories: [poster]
tags: brain_decoding word2vec dependency_based_word_embedding distibutional_semantics cognitive_plausibility
comments: true
mathjax: true
excerpt: Our paper “Experiential, Distributional and Dependency-based Word Embeddings have Complementary Roles in Decoding Brain Activity” has been accepted for poster presentation at Cognitive Modeling and Computational Linguistics workshop (2018).
image:
  feature: posters/poster_final.png
---

![](posters/poster_final.png)

Checkout our [paper](https://www.aclweb.org/anthology/W18-0107/), {% cite abnar-etal-2018-experiential %}, and the [codes](https://github.com/samiraabnar/NeuroSemantics) to reproduce our experiments!

>
We evaluate 8 different word embedding models on their usefulness for predicting the neural activation patterns associated with concrete nouns. The models we consider include an experiential model, based on crowd-sourced association data, several popular neural and distributional models, and a model that reflects the syntactic context of words (based on dependency parses). Our goal is to assess the cognitive plausibility of these various embedding models, and understand how we can further improve our methods for interpreting brain imaging data.
We show that neural word embedding models exhibit superior performance on the tasks we consider, beating experiential word representation model.The syntactically informed model gives the overall best performance when predicting brain activation patterns from word embeddings; whereas the GloVe distributional method gives the overall best performance when predicting in the reverse direction (words vectors from brain images). Interestingly, however, the error patterns of these different models are markedly different. This may support the idea that the brain uses different systems for processing different kinds of words. Moreover, we suggest that taking the relative strengths of different embedding models into account will lead to better models of the brain activity associated with words.
