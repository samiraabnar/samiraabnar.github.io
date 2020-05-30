---
layout: post
title:  "Quantifying Attention Flow in Transformers"
date:   2020-04-05 23:00
categories: [blogpost]
tags: transformer self_attention attention_visualisation attention_rollout attention_flow bert subject_verb_agreement language_modelling
comments: true
mathjax: true
excerpt: In this post, I explain two techniques for visualising attention that address the problem of lack of token identifiability in higher layers of Transformers when using raw attention weights to interpret models' decisions. These techniques are called <kbd>Attention Rollout</kbd> and <kbd>Attention Flow</kbd> that are introduced in our paper "<a href='https://arxiv.org/abs/2005.00928'>Quantifying Attention Flow In Transformers</a>".
image:
  feature: flow_images/attention_flow.gif
  twitter: flow_images/bert_example.png
---

<!-- we show that compared to raw attention weights, the token attentions from <kbd>Attention Rollout</kbd> and <kbd>Attention Flow</kbd> have higher correlations with the importance scores obtained from input gradients as well as an input ablation based attribution method. Furthermore, we visualise the token attention weights and demonstrate that they are better approximations of how input tokens contribute to a predicted output, compared to raw attention weights. -->

Attention has become the key building block of neural sequence processing models,
and visualising attention weights is the easiest and most popular approach to interpret a model's decisions and to gain insights about its internals.
Although it is wrong to equate attention with explanation {% cite pruthi2019learning jain2019attention %}, it can still offer plausible and meaningful interpretations {% cite wiegreffe2019attention vashishth2019attention vig2019visualizing %}.
In this post, I focus on problems arising when we move to the higher layers of a model, due to lack of token identifiability of the embeddings in higher layers {% cite brunner2019validity %}. I discuss the ideas proposed in [our paper](https://arxiv.org/abs/2005.00928) for visualising and interpreting attention weights taking this problem into account!

<!--more-->

Here, I explain two simple but effective methods, called **Attention Rollout** and **Attention Flow**, to compute attention scores to input tokens  (i.e., _token attention_) at each layer, by taking raw attentions (i.e., _embedding attention_) of that layer as well as those from the precedent layers.

Let's first discuss the token identifiability problem in Transformers in more details.

##### Attention to Embeddings vs Attention to Input Tokens
In the Transformer model, in each layer, _self-attention_ combines information from attended embeddings of the previous layer to compute new embeddings for each token. Thus, across layers of the Transformer, information originating from different tokens gets increasingly mixed (Check out the {% cite brunner2019validity %} for a more thorough discussion on how the identity of tokens get less and less represented in the embedding of that position as we go into deeper layers.).

Hence, when looking at the $i$th self-attention layer, we can not interpret the attention weights as the attention to the input tokens, i.e., embeddings in the input layer. This makes attention weights unreliable as explanation probes to answer questions like "Which part of the input is the most important when generating the output?" (except for the very first layer where the self-attention is directly applied to the input tokens.)
<!-- ![Raw Attention Weights](img/flow_images/rat_deep_1.png){:style="height: 360px; float: right"} -->

{% include image.html
            img="img/flow_images/rat_deep_1.png"
            title=""
            caption="Raw attention weights"
            style="height: 400px; display: block; float:right; padding:0px; margin-bottom:40px; margin-top:0px; margin-right:1px"
            height="400px" %}

Let's take a look at the example in the figure that shows how attention weights in a Transformer model change across layers. In this figure, we see the attention weights of a 6-layer Transformer encoder trained on the subject-verb agreement classification task for an example sentence.
In the subject-verb agreement task, given a sentence up to its verb, the goal is to classify the number of the verb. To be able to do this, a model needs to recognise the subject of that verb correctly. For the example in the figure, <kbd>The key to the cabinets &lt;verb&gt;</kbd>, intuitively we expect the model to attend to the token <kbd>key</kbd>, which is the subject of the missing verb, to classify the verb number correctly. Or the token <kbd>cabinets</kbd>, the attractor in case it is making a mistake.

However, if we only look at the attention weights in the last layer, it seems all input tokens have more or less equal contributions to the output of the model since the attention weights from the <kbd>CLS</kbd> token in this layer are almost uniformly distributed over all embeddings. But if we also take into account the attention weights in the previous layers, we realise that some of the input tokens are getting more attention in earlier layers. Notably, in layer 1, the embedding for the verb is mostly attending to the token <kbd>key</kbd>, while in the third layer, the <kbd>CLS</kbd> token is mostly attending to the embedding of the verb.

So, if we want to use attention weights to understand how a self-attention network works, we need to take the flow of information in the network into account. One way to do this is to use attention weights to approximate the information flow while taking different aspects of the architecture of the model into account, e.g., how multiple heads interact or the residual connections.

##### Information Flow Graph of a Transformer Encoder
Let's take a look at the schematic view of self-attention layer in the Transformer Model introduced in {% cite vaswani2017attention %}(figure below):
{% include image.html
            img="img/flow_images/attention_block.png"
            title=""
            caption="Transformer encoder block"
            style="height: 300px; display: block; float:left"
            height="300px" %}

Given this attention module with residual connections, we compute values in layer $l+1$ as $V_{l+1} = V_{l}  + W_{att}V_l$, where $ W_{att}$ is the attention matrix. Thus, we have $V_{l+1} = (W_{att} + I) V_{l}$. So, to account for residual connections, we add an identity matrix to the attention matrix and re-normalize the weights. This results in $A = 0.5W_{att} + 0.5I$, where $A$ is the raw attention updated by residual connections.

We can create the information flow graph of a Transformer model, using this equation as an approximation of how information propagates in the self-attention layers. Using this graph, we can take the attention weights in all layers into account and translate the attention weights in each layer to attention to input tokens.

We can model the information flow in the network with a [_DAG_ (Directed Acyclic Graph)](https://en.wikipedia.org/wiki/Directed_acyclic_graph), in which input tokens and hidden embeddings are the nodes, edges are the attentions from the nodes in each layer to those in the previous layer, and the weights of the edges are the attention weights.
Note that, we augment this graph with residual connections to more accurately model the connections between input tokens and hidden embeddings.


##### From Attention to Embeddings to Attention to Tokens
Given this graph, based on how we interpret the weights associated with the edges, which are the raw attention weights, we can use different techniques to compute the attention from each node in the graph to the input tokens.

###### Attention Rollout

Assume the attention weights determine the proportion of the incoming information that can propagate through each link, i.e., the identities of input tokens are linearly combined through the layers based on the attention weights. Then, to compute the attention to input tokens in layer $$i$$ given all the attention weight in the previous layers, we recursively multiply the attention weights matrices, starting from the input layer up to layer $$i$$.
In the figure below, we show how Attention Rollout works in a simple attention DAG. In this example, the goal is to compute the attention from the embedding of the last position in the last layer to the first input token. We see that the attention weights in the second layer are multiplied by the attention weights from the first layer to compute the final attention score.

<!-- <div style="width: 700; display:inline-block; clear: right; vertical-align:middle;"> -->
{% include image.html
            img="img/flow_images/attention_rollout.gif"
            title=""
            caption="Attention Rollout"
            width="360px" %}


<!-- </div> -->

###### Attention Flow
If we view the attention weights as the capacity of each link, the problem of computing the attention in layer $$i$$ to the input tokens reduces to the [maximum flow problem](https://en.wikipedia.org/wiki/Maximum_flow_problem), where we want to find the maximum flow value from each input token to each position in layer $$i$$.  

In the figure below, we see the same example like the one we saw for the Attention Rollout, except here the attention weights are viewed as the capacity of the edges. Thus, the total attention score of a path is the smallest capacity of the edges in that path. This is a straightforward example, and maximum computing flow can be more complicated when paths overlap.

{% include image.html
            img="img/flow_images/attention_flow.gif"
            title=""
            caption="Attention Flow"
            width="360px" %}
##### How does this all work in practice?

Let's see an example of how these techniques work in practice!
Applying these techniques to a pretrained 24-layer BERT model, we get some insights on how the models resolve pronouns.
What we do here is to feed the model with a sentence, masking a pronoun. Next, we look at the prediction of the model for the masked pronoun and compare the probabilities predicted for <kbd>her</kbd> and <kbd>his</kbd>.

{% include image.html
            img="img/flow_images/bert_example.png"
            title=""
            caption="Visualing attention for a 24-layer BERT"
            width="360px" %}


As we can see, in the first example (figure a), the prediction of the model is <kbd>his</kbd>. Hence, we expect the model to attend to the word <kbd>author</kbd> rather than <kbd>Sara</kbd>.
In this case, both Attention Rollout and Attention Flow are consistent with this intuition.
Whereas, the final layer of Raw Attention does not seem to be consistent with the prediction of the models, and it varies a lot across different layers.

In the second example, the prediction of the model is <kbd>her</kbd>, hence, we expect the model to pay more attention to the word <kbd>Mary</kbd>. However, both Raw Attention weights and Attention Rollout show that the model is attending to <kbd>John</kbd>. In this case, only Attention Flow weights are consistent with our intuition and the prediction of the model.
In some sense, Attention Rollout is more restrictive compared to Attention Flow, and it provides us with more exaggerated differences between attention scores to different input tokens (because it multiplies the weights). This can be a source of error for Attention Rollout considering the approximations we have in these techniques.

Note that, both Attention Rollout and Attention Flow are  **post hoc methods for visualisation and interpretation purposes** and they do not provide new attention weights to be used during training or inference.

To see more examples, you can try out [this notebook](https://github.com/samiraabnar/attention_flow/blob/master/bert_example.ipynb). And for more details, such as how we can handle multiple heads take a look at our paper, "[Quantifying Attention Flow In Transformers][2ae63ee1]".



  [2ae63ee1]: https://arxiv.org/abs/2005.00928 "Quantifying Attention Flow In Transformers"
