---
layout: home
title: Home
permalink: /home/
---
<!-- Just redirect to home.md -->
{% assign p = site.pages | where: 'name','index.md' %}
{{p}}
