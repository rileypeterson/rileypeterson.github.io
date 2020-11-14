---
layout: project
title:  "NFL Matchup Modeling"
date:   2020-11-06
categories: jekyll update
preview-image: preview.jpg
defaults:
  - scope:
    path: "assets/img"
---

---
### Introduction
The NFL consists of 16 regular season games. Each week teams battle it out on the field while oddsmakers in Vegas
are determined to set lines that will be bet equally on both sides. Traditionally, the three most common betting
lines are:

-  Moneyline - The team that will win the game.
-  Spread - The differential between the favored team's score and the underdog.
-  Over/Under - The total number of points in the game.

As an example, here are the current lines for an upcoming game this weekend (Week 10):

{{ page.title | downcase | slugify }}/images/game-lines.png)

![Example Game Lines](/game-lines.png)
{: class="project-image"}

The goal of this project is to use mathematical methods and analysis to correctly predict the winning side, the spread, and total points.

---

### Methodology

#### Points Prediction: Convolved PDFs
The idea behind this approach is simple (primitive). Let's say we would like to know the number of points team 1
will score against team 2. Consider the distribution of points scored by the team 1 offense
in preceding weeks. Consider the distribution of points surrendered by the team 2 defense in preceding
 weeks. Sticking with the matchup above:


| Week                              |   1 |   2 |   3 |   4 |   5 |   6 |   7 | 8   |   9 |
|-----------------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Houston Texans's Points Scored    |  20 |  16 |  21 |  23 |  30 |  36 |  20 | BYE |  27 |
| Houston Texans's Points Surrended |  34 |  33 |  28 |  31 |  14 |  42 |  35 | BYE |  25 |


{% capture details %}
{% highlight python %}
{% include_relative snippets/score-scrape.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Score Scraper
{% endcapture %}
{% include snippet.html %}

This is just a distribution of numbers so from its mean and standard deviation
we can form a gaussian:

![Example Gaussian]({{ page.title | downcase | slugify }}/images/game-lines.png)
{: class="project-image"}

{% capture details %}
{% highlight python %}
{% include_relative snippets/pdfs-graphs.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Offensive Scores Gaussian
{% endcapture %}
{% include snippet.html %}

Now consider the distribution of points surrendered by the Cleveland Browns defense:

|                                  |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 | 9   |
|----------------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Cleveland Browns's Points Scored |  38 |  30 |  20 |  38 |  23 |  38 |  34 |  16 | BYE |

{% capture details %}
{% highlight python %}
{% include_relative snippets/score-scrape-browns-defense.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Score Scraper (Browns Defense)
{% endcapture %}
{% include snippet.html %}

![Example Gaussian]({{ page.title | downcase | slugify }}/images/game-lines.png)
{: class="project-image"}

{% capture details %}
{% highlight python %}
{% include_relative snippets/pdfs-graphs.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Offensive Scores Gaussian
{% endcapture %}
{% include snippet.html %}
