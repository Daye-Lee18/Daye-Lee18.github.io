---
layout: page
title: Projects
permalink: /projects/
description: A collection of my projects.
nav: true
nav_order: 3
display_categories: [master's degree thesis, work, study, fun]
---

<div class="projects-page">
  <p class="projects-intro">
    Over the last few years, I have worked on several research and development projects.
    You can find an overview of selected projects here.
  </p>

{% if site.enable_project_categories and page.display_categories %}
{% for category in page.display_categories %}
<a id="{{ category }}" href=".#{{ category }}">

<h2 class="project-category">{{ category | capitalize }}</h2>
</a>

      {% assign categorized_projects = site.projects | where: "category", category | sort: "importance" %}

      <div class="project-list">
        {% for project in categorized_projects %}
          {% include project_featured.liquid %}
        {% endfor %}
      </div>
    {% endfor %}

{% else %}
{% assign sorted_projects = site.projects | sort: "importance" %}

<div class="project-list">
{% for project in sorted_projects %}
{% include project_featured.liquid %}
{% endfor %}
</div>
{% endif %}

</div>
