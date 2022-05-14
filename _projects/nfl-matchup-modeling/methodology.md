## Points Prediction: Convolved PDFs
The idea behind this approach is simple (dumb). Let's say we would like to know the number of points team 1
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

| Week                             |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 | 9   |
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