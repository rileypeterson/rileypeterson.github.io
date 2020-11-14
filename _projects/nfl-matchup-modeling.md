---
layout: project
title:  "NFL Matchup Modeling"
date:   2020-11-06
categories: jekyll update
preview-image: preview.jpg
---

{%- assign slug-name = page.title | slugify | downcase -%}
---
### Introduction
The NFL consists of 16 regular season games. Each week teams battle it out on the field while oddsmakers in Vegas
are determined to set lines that will be bet equally on both sides. Traditionally, the three most common betting
lines are:

-  Moneyline - The team that will win the game.
-  Spread - The differential between the favored team's score and the underdog.
-  Over/Under - The total number of points in the game.

As an example, here are the current lines for an upcoming game this weekend (Week 10):


![Example Game Lines](/assets/images/{{ slug-name }}/game-lines.png)
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
{% include_relative snippets/{{ slug-name }}/score-scrape.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Score Scraper
{% endcapture %}
{% include snippet.html %}


This is just a distribution of numbers so from its mean and standard deviation
we can form a Gaussian:

![Example Gaussian](/assets/images/{{ slug-name }}/gaussian1.png)
{: class="project-image"}

{% capture details %}
{% highlight python %}
{% include_relative snippets/{{ slug-name }}/pdf-graphs.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Offensive Points Gaussian PDF
{% endcapture %}
{% include snippet.html %}


Now consider the distribution of points surrendered by the Cleveland Browns defense:

|                                  |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 | 9   |
|----------------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Cleveland Browns's Points Scored |  38 |  30 |  20 |  38 |  23 |  38 |  34 |  16 | BYE |


{% capture details %}
{% highlight python %}
{% include_relative snippets/{{ slug-name }}/score-scrape-browns-defense.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Score Scraper (Browns Defense)
{% endcapture %}
{% include snippet.html %}

Combining these together we get the following Gaussian distributions:

![Matchup Distribution](/assets/images/{{ slug-name }}/pdf-graphs-with-convolution.png)
{: class="project-image"}

{% capture details %}
{% highlight python %}
{% include_relative snippets/{{ slug-name }}/pdf-graphs-with-convolution.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Matchup Distribution
{% endcapture %}
{% include snippet.html %}

For those who are not familiar with [convolutions](https://en.wikipedia.org/wiki/Convolution), it essentially will morph 
two waveforms together. This means that should a dominant offense (30 points per game (ppg)) play a weak defense (surrenders 40 ppg), the resulting 
waveform will predict the offense scoring more than usual (~35 points). However, if they play a strong defense (surrenders 20 ppg), then the expected offensive score will be lower (somewhere between ~25 points). If they play a defense which typically surrenders 30 ppg, then the convolution will 
support this expectation.

![Convolution](/assets/images/{{ slug-name }}/convolution.png)
{: class="project-image"}

{% capture details %}
{% highlight python %}
{% include_relative snippets/{{ slug-name }}/convolution.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
Convolution
{% endcapture %}
{% include snippet.html %}

So this expects the Houston Texans to score 27 +/- 5 points against the Cleveland Browns this weekend. 

---

#### Shortcomings
If you know anything about the upcoming matchup I've been running with you would think that's crazy. Here are the reasons why:

*  First and foremost:
    > The Weather Channel is calling for an 80 percent chance of rain in Cleveland on Sunday, with 25-35 mph winds and occasional gusts over 50 mph.

    Clearly this model does not account for inclement weather.
*  Large standard deviations. If the distribution of points is "too wide", then it has trouble predicting a blowout more one sided result.
*  No recency bias. The Colts have come a long way since their week 1 woes. What you showed me last week matters much more than week one.
*  Personnel considerations. The Cowboys lost Dak and Dalton over the past few weeks. Jerry is picking guys off the street to fill their QB spot. Big Ben might be out with COVID.
*  It doesn't account for the strength of the defenses you played. The way to the think about this is imagine t1 which has only played lousy defenses
they put up 40 points a game. Their offense isn't that good, they've just been playing high school defenses. Then they go to Pittsburgh and play the hard nosed Steelers, even if the model is saying they will put up less points than usual, they will probably put up **much** less because they haven't played a real defense. I think this is the most accessible aspect to address, just takes some careful thought.
*  Doesn't work for no prior data. Guess I'm sticking to my gut for week 1 picks next year. 😩

---

Last time it was windy in Cleveland 😂:

![WINDY](https://user-images.githubusercontent.com/29719483/99143552-2d3ae280-2613-11eb-82e9-941a41ad5a29.gif)
{: class="project-image"}
