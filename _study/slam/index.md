---
layout: page
title: SLAM
description: A step-by-step study of the history of SLAM and of LiDAR-inertial state estimation.
permalink: /study/slam/
topic_index: true
---

This SLAM study is organised in two tracks. History first covers how the problem is defined and how the algorithms changed; LIO then compares FAST-LIO2, which is deployed on Vision60, with related systems.

<div class="study-section-list">
  <a class="study-section-card" href="{{ '/study/slam/history/' | relative_url }}">
    <span class="study-section-kicker">Part 1 · 12 chapters</span>
    <strong>SLAM History</strong>
    <span>Short chapters tracing the flow from probabilistic SLAM through Graph SLAM, Visual SLAM, LIO and Spatial AI.</span>
  </a>
  <a class="study-section-card" href="{{ '/study/slam/state-estimation/' | relative_url }}">
    <span class="study-section-kicker">Part 2 · paper reviews</span>
    <strong>LIO &amp; State Estimation</strong>
    <span>Using FAST-LIO2 as the reference point, compares the sensor-fusion architectures and papers of LIO-SAM, FAST-LIVO2, VILENS and others.</span>
  </a>
  <a class="study-section-card" href="{{ '/study/slam/leg-fusion/' | relative_url }}">
    <span class="study-section-kicker">Part 3 · project notes</span>
    <strong>Leg-Velocity Fusion in FAST-LIO2</strong>
    <span>다리 운동학 속도를 FAST-LIO2 필터에 관측으로 추가하는 구조와 계산 예제. (한국어)</span>
  </a>
</div>

## Suggested study order

1. Chapters 1–3 of History explain the SLAM problem and the filtering family.
2. Chapters 4–7 connect 3D rotation, ICP, factor graphs and sparse optimisation.
3. Chapters 8–12 cover sensors, map representations, LIO, learning-based methods and evaluation.
4. LIO & State Estimation organises the architecture of each paper of interest and the conditions for applying it on a real robot.

Downloaded public slides and papers are kept locally in `_resource/slam`; sources and original links are recorded in the [resource index](https://github.com/Daye-Lee18/Daye-Lee18.github.io/tree/main/_resource/slam).
