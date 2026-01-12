---
layout: page
title: Dance
description: Amateur Dancer at SNU
img: assets/img/thumnail/goahead.png
importance: 1
category: fun
videos:
  - id: qi2RQLVIZ00?si
    title: THIQUE - Jojo Gomez Choreo
    channel: GoAheaD
    year: 2025
    url: https://youtube.com/watch?v=qi2RQLVIZ00
  - id: X3l6_GUB8q0
    title: SPORTS CAR + It’s ok I’m ok - Jojo Gomez Choreo
    channel: GoAheadD
    year: 2025
    url: https://youtube.com/watch?v=X3l6_GUB8q0
  - id: urly6X8rKXo
    title: GIMME MORE + Outrageous - Saarah Fernandex + Jojo Gomez Choreo
    channel: GoAheaD
    year: 2023
    url: https://www.youtube.com/watch?v=urly6X8rKXo
  - id: tmSDNdcOZNQ
    title: Where Have You Been - BELEGACY Choreo
    channel: GoAheaD
    year: 2022
    url: https://youtube.com/watch?v=tmSDNdcOZNQ
  - id: UNakiI-evF8
    title: Throw A Fit - Jojo Gomez Choreo
    channel: GoAheaD
    year: 2022
    url: https://youtube.com/watch?v=UNakiI-evF8
  - id: 3sstIYiOVuU
    title: Feeling good - Michael Buble + Buttons - The Pussycat Dolls
    channel: GoAheaD
    year: 2022
    url: https://youtube.com/watch?v=3sstIYiOVuU
  - id: If3rRKm_R3k
    title: 2-H.E.R & Mantra-Troyboi
    channel: GoAheaD
    year: 2019
    url: https://youtube.com/watch?v=If3rRKm_R3k
  - id: 6dcF3ucnZ84
    title: I Got You (Jessie J) + Dangerous Woman (Ariana Grande)
    channel: IEtudel
    year: 2019
    url: https://youtube.com/watch?v=6dcF3ucnZ84
  - id: mHpdDDsDjjw
    title: End of time remix - Beyonce + Doctor Pepper (Shintaro remix) - CL x Diplo x RIFF RAFF x OG Maco (Waack of the world in feedback competition 4)
    channel: GoAheaD
    year: 2019
    url: https://youtube.com/watch?v=mHpdDDsDjjw
  - id: 1rw03e7t5nY
    title: Did It On 'Em + Barbie Tingz (Nicki Minaj) Filmed by lEtudel
    channel: GoAheaD
    year: 2018
    url: https://youtube.com/watch?v=1rw03e7t5nY
  - id: acKAzPIaIWE
    title: Love so soft - Kelly Clarkson (whatdowwari) Filmed by lEtudel
    channel: GoAheaD
    year: 2018
    url: https://youtube.com/watch?v=acKAzPIaIWE
---

<p>참여한 공영 영상들을 나열하였습니다. 더보기 란에 제 이름이 가장 먼저 오는 곡은 제가 곡의 총괄로서, 무대 구성, 의상, 부분 안무 창작, 음악 편집 및 조명 구성등의 공연 관련한 대부분의 작업을 담당하였습니다. 대학교 2학년때부터 참여한 동아리에서의 결과물들을 모으다보니 감회가 남다르네요!🔥 </p>

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
