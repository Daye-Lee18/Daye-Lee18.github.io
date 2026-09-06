---
layout: page
title: SLAM
description: SLAM의 역사와 LiDAR-inertial state estimation을 단계별로 공부합니다.
permalink: /study/slam/
topic_index: true
---

SLAM 학습은 두 갈래로 구성했습니다. 먼저 History에서 문제의 정의와 알고리즘의 변화를 익힌 뒤, LIO에서 Vision60에 적용한 FAST-LIO2와 관련 시스템을 비교합니다.

<div class="study-section-list">
  <a class="study-section-card" href="{{ '/study/slam/history/' | relative_url }}">
    <span class="study-section-kicker">Part 1 · 12 chapters</span>
    <strong>SLAM History</strong>
    <span>확률적 SLAM부터 Graph SLAM, Visual SLAM, LIO, Spatial AI까지 흐름을 짧은 장으로 학습합니다.</span>
  </a>
  <a class="study-section-card" href="{{ '/study/slam/lio/' | relative_url }}">
    <span class="study-section-kicker">Part 2 · paper reviews</span>
    <strong>LIO &amp; State Estimation</strong>
    <span>FAST-LIO2를 기준으로 LIO-SAM, FAST-LIVO2, VILENS 등 센서 융합 구조와 논문을 비교합니다.</span>
  </a>
</div>

## 추천 학습 순서

1. History 1~3장에서 SLAM 문제와 filtering 계열을 이해합니다.
2. 4~7장에서 3D 회전, ICP, factor graph와 희소 최적화를 연결합니다.
3. 8~12장에서 센서·지도 표현·LIO·학습 기반 방법과 평가를 살펴봅니다.
4. LIO & State Estimation에서 관심 논문의 구조와 실제 로봇 적용 조건을 정리합니다.

다운로드한 공개 슬라이드와 논문은 로컬 `_resource/slam`에 보관하고, 출처와 원문 링크는 [자료 색인](https://github.com/Daye-Lee18/Daye-Lee18.github.io/tree/main/_resource/slam)에 기록합니다.
