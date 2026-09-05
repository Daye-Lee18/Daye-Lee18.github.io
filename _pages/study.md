---
layout: page
title: Study
permalink: /study/
description: Notes and materials I'm working through.
nav: true
nav_order: 4.5
display_categories: [IsaacSim, SLAM, HW, RL, RUST]   # ←주제(토픽) 목록
---


<div class="study-page">
  <p class="study-intro">
    The subjects I've studied throughout the course of my life.
  </p>

{% if page.display_categories %}
{% for category in page.display_categories %}
<a id="{{ category }}" href=".#{{ category }}">

<h2 class="project-category">{{ category }}</h2>
</a>

      {% assign categorized_study = site.study | where: "category", category | sort: "importance" %}

      <div class="project-list">
        {% for project in categorized_study %}
          {% include project_featured.liquid %}
        {% endfor %}
      </div>
    {% endfor %}

{% else %}
{% assign sorted_study = site.study | sort: "importance" %}

<div class="project-list">
{% for project in sorted_study %}
{% include project_featured.liquid %}
{% endfor %}
</div>
{% endif %}

</div>
