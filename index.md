---
layout: home
title: Home
permalink: /
---
<!-- Just redirect to home.md -->
{% assign p = site.pages | where: 'name','home.md' %}
{{p}}