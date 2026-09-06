---
layout: page
title: Study
permalink: /study/
description: Notes and materials I'm working through.
nav: true
nav_order: 4.5
display_categories: [IsaacSim, SLAM, EdgeComputing, RL, RUST] # ←주제(토픽) 목록
---

<div class="study-page">
  <p class="study-intro">
    The subjects I've studied throughout the course of my life.
  </p>

  <h2>Study Notes</h2>

  <div class="study-card-list">
    {% for category in page.display_categories %}
      {% assign topic = site.study | where: "category", category | where: "topic_index", true | first %}
      {% if topic %}
        {% include study_card.liquid %}
      {% endif %}
    {% endfor %}
  </div>

  <h2 class="study-resource-heading">Other Study Resources</h2>

  <div class="study-card-list">
    {% assign study_resources = site.data.study_resources | sort: "importance" %}
    {% for resource in study_resources %}
      {% include study_resource_card.liquid resource=resource %}
    {% endfor %}
  </div>
</div>
