---
layout: post
title:  "The Contextual Bandit"
date:   2020-12-25
categories: interesting
---

Merry Christmas!! I just found this to be an interesting definition.

According to [this paper](https://arxiv.org/pdf/1802.09127.pdf):

> The contextual bandit problem works as follows. At time $$ t = 1, . . . , n $$ a new context $$X_t \in R^d$$
arrives and is presented to algorithm $$\mathcal{A}$$. The algorithm —based on its internal model and $$X_t$$— selects
one of the $$k$$ available actions, $$a_t$$. Some reward $$r_t = r_t(X_t, a_t)$$ is then generated and returned to the
algorithm, that may update its internal model with the new data. At the end of the process, the reward
for the algorithm is given by $$r = \sum_{t=1}^{n} r_t$$, and cumulative regret is defined as 
$$R_\mathcal{A} = \mathop{\mathbb{E}}[r^* - r]$$ − r], where $$r^*$$ is the cumulative reward of 
the optimal policy (i.e., the policy that always selects the action
with highest expected reward given the context). The goal is to minimize $$R_\mathcal{A}$$.  


{% capture details %}
<pre>
<code class="citation">@misc{riquelme2018deep,
      title={Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling},  
      author={Carlos Riquelme and George Tucker and Jasper Snoek},  
      year={2018},  
      eprint={1802.09127},  
      archivePrefix={arXiv},  
      primaryClass={stat.ML}  
}
</code>
</pre>
{% endcapture %}
{% capture title %}
Citation
{% endcapture %}
{% include snippet.html %}