---
layout: post
title:  "Fixed Point Deming Regression"
date:   2021-02-11
categories: interesting, math
---

# Introduction

Traditionally, linear/polynomial regression approaches assume that all the error is due to the dependent variable. So in the linear case the objective is to minimize the following (simple linear regression):

$$ S = \sum_{i=1}^{N} (y_i - (mx_i + b))^2$$

The independent variable, x is assumed factual (i.e. it has no error). This is often true. Here's a picture of what this looks like:

![Simple](/assets/images/posts/2021-02-11-fixed-point-deming-regression/simple_least_squares.png)
{% capture details %}
{% highlight python %}
{% include_relative snippets/2021-02-11-fixed-point-deming-regression/simple_least_squares.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
{% endcapture %}
{% include snippet.html %}


However, you can also assume that x and y have some equivalent error contribution. This is known as [Deming Regression](https://en.wikipedia.org/wiki/Deming_regression), where you try to minimize the orthogonal distance between the line of best fit and the (x, y) points. Here's a picture (Wikipedia):


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png" style="background: white; margin-left: auto; margin-right: auto; display: block; width: 40%;">

For my application, I wanted to know how find the Deming regression line <b><u>when the intercept is fixed at 0 (i.e. b=0)</u></b>. This amounts to minimizing the following (fixed point deming regression):

$$ S = \sum_{i=1}^{N} [(y_i - mx_i^*)^2 + (x_i - x_i^*)^2]$$


I looked all over for this and couldn't find a derivation or result. I worked it out and I thought I would share the result. This is probably one of the easier problems in a calculus textbook somewhere, so nothing crazy. :)

Here we go (find $$x_i^*$$ which are optimal):

$$ S = \sum_{i=1}^{N} [(y_i - mx_i^*)^2 + (x_i - x_i^*)^2]$$

$$ \frac {\partial S}{\partial x_i^*} = \sum_{i=1}^{N} [-2m(y_i - mx_i^*) - 2(x_i - x_i^*)] = 0$$

$$ \sum_{i=1}^{N} [m(y_i - mx_i^*) + (x_i - x_i^*)] = 0$$

$$ \sum_{i=1}^{N} [m y_i - m^2 x_i^* + x_i - x_i^*] = 0$$

$$ -(1 + m^2) \sum_{i=1}^{N} x_i^* + \sum_{i=1}^{N} (m y_i + x_i) = 0$$

$$ 
\begin {equation} 
\implies \sum_{i=1}^{N} x_i^* = \sum_{i=1}^{N} \frac{m y_i + x_i}{1 + m^2} \tag{1}\label{eq:one}
\end{equation}
$$

Now find the partial with respect to the slope $$m$$:

$$ \frac {\partial S}{\partial m} = \sum_{i=1}^{N} [2x_i^*(y_i - mx_i^*)] = 0$$

$$ 
\begin {equation} 
\implies m = \frac {\sum_{i=1}^{N} x_i^* y_i}{ \sum_{i=1}^{N} x_i^{*^2}} \tag{2}\label{eq:two}
\end{equation}
$$


Plugging in \eqref{eq:one} to \eqref{eq:two} begets:

$$ m = \frac {\sum_{i=1}^{N} (\frac{m y_i + x_i}{1 + m^2}) y_i}{ \sum_{i=1}^{N} \left(\frac{m y_i + x_i}{1 + m^2}\right)^2} $$

$$ m \sum_{i=1}^{N} {\left(\frac{m y_i + x_i}{1 + m^2}\right)}^2 = \sum_{i=1}^{N} \left(\frac{m y_i + x_i}{1 + m^2}\right) y_i $$

$$ m \sum_{i=1}^{N} {\left(m y_i + x_i\right)}^2 = \sum_{i=1}^{N} \left(m y_i + x_i + m^3y_i + m^2 x_i\right) y_i $$

$$ \sum_{i=1}^{N} \left(\xcancel{m^3 y_i^2} + \xcancel{2}m^2 x_i y_i + m x_i^2 \right) = 
   \sum_{i=1}^{N} \left(m y_i^2 + x_i y_i + \xcancel{m^3y_i^2} + \xcancel{m^2 x_i y_i}\right)
$$

$$ m^2 \sum_{i=1}^{N} x_i y_i + m \sum_{i=1}^{N} (x_i^2 - y_i^2) + \sum_{i=1}^{N} (-x_i y_i) = 0 $$

We have:

$$ a = \sum_{i=1}^{N} x_i y_i, \quad b = \sum_{i=1}^{N} (x_i^2 - y_i^2), \quad c = \sum_{i=1}^{N} (-x_i y_i)$$

$$ \implies m = \frac {-b \pm \sqrt{b^2 - 4 a c}}{2a} $$

These yields two values for $$m$$. There's probably some official way to discern the true value of $$m$$, but I'm lazy and 
just pick the one which minimizes $$S$$ in our original equation. See the code below.

---
# Result

![Deming](/assets/images/posts/2021-02-11-fixed-point-deming-regression/deming_least_squares.png)
{% capture details %}
{% highlight python %}
{% include_relative snippets/2021-02-11-fixed-point-deming-regression/deming_least_squares.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
{% endcapture %}
{% include snippet.html %}



:-------------------------:|:-------------------------:
![Simple](/assets/images/posts/2021-02-11-fixed-point-deming-regression/simple_least_squares.png)  |  ![Deming](/assets/images/posts/2021-02-11-fixed-point-deming-regression/deming_least_squares.png)


![Deming](/assets/images/posts/2021-02-11-fixed-point-deming-regression/vs_least_squares.png)
{% capture details %}
{% highlight python %}
{% include_relative snippets/2021-02-11-fixed-point-deming-regression/vs_least_squares.py %}
{% endhighlight %}
{% endcapture %}
{% capture title %}
{% endcapture %}
{% include snippet.html %}