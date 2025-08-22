---
layout: page
title: TED talk
description: English to Korean translation volunteer
img: assets/img/thumnail/ted.jpeg
importance: 1
category: fun
videos:
  - id: T1aZxcyiYAw
    title: Why did it take so long to find giant squids? 
    channel: TED-Ed
    year: 2024
    url: https://www.youtube.com/watch?v=T1aZxcyiYAw
  - id: CeUoS2T2hhc
    title: What’s the Future of Food? A Chef + a Cardiologist Answer 
    channel: TEDx
    year: 2023
    url: https://www.youtube.com/watch?v=CeUoS2T2hhc
---

<p>번역 작업을 한 영상들 리스트를 나열하였습니다.</p>

<div class="ott-grid">
  {% for v in page.videos %}
    <a class="ott-card" href="{{ v.url }}" target="_blank" rel="noopener">
      <div class="thumb">
        <img src="https://img.youtube.com/vi/{{ v.id }}/hqdefault.jpg" alt="{{ v.title }}">
      </div>
      <div class="meta">
        <h3 class="title">{{ v.title }}</h3>
        <p class="sub">{{ v.channel }} · {{ v.year }}</p>
      </div>
    </a>
  {% endfor %}
</div>

