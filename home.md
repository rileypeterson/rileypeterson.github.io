---
layout: home
title: Home
permalink: /home/
---
<hr>
<!-- Just redirect to home.md -->
{% assign p = site.pages | where: 'name','about.md' %}
{{p}}
![Me](assets/images/photos/banff.jpg)