---
layout: post
title: Incremental Reading for Question Answering
date:   2018-12-12 23:00
categories: [poster]
tags: lstm question_answering incremental_reading early_stopping cognitive_plausibility
comments: true
mathjax: true
excerpt: We presented our paper on “Incremental Reading for Questions Answering” in the Continual Learning workshop at NeurIPS 2018 in Montreal. This is about the project I worked on during my internship at Google in summer 2018.
image:
  feature: posters/CL_workshop_IncReading_forPrint.png
---

![](posters/CL_workshop_IncReading_forPrint.png)

Checkout our [paper](https://arxiv.org/abs/1901.04936) {% cite Abnar2019IncrementalRF %}!

>Any system which performs goal-directed continual learning must not only learn incrementally but process and absorb information incrementally. Such a system also has to understand when its goals have been achieved. In this paper, we consider these issues in the context of question answering. Current state-of-the-art question answering models reason over an entire passage, not incrementally. As we will show, naive approaches to incremental reading, such as restriction to unidirectional language models in the model, perform poorly. We present extensions to the DocQA model to allow incremental reading without loss of accuracy. The model also jointly learns to provide the best answer given the text that is seen so far and predict whether this best-so-far answer is sufficient.
