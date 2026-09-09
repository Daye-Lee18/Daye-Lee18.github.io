---
layout: study-series
title: "Leg-Velocity Fusion in FAST-LIO2"
description: 다리 운동학 속도를 FAST-LIO2 필터에 관측으로 추가하는 구조와, 그것을 설명하기 위한 계산 예제.
category: SLAM
series: leg_fusion
series_index: true
importance: 3
permalink: /study/slam/leg-fusion/
---

FAST-LIO2에 다리 속도를 넣는 작업의 **구조**와 **수치 예제**를 정리한다.
다른 사람에게 설명할 때 쓸 수 있도록, 개념마다 손으로 따라갈 수 있는 문제를 붙였다.

```text
Ch 1   구조 — 어디에 무엇이 들어가나        두 SLAM 이 아니라 관측 하나 추가
Ch 2   계산 예제 — 문제로 이해하기          좌표 변환 · 칼만 갱신 · chi2 gate
```

> 관련 논문 리뷰: [VILENS]({{ '/study/slam/state-estimation/vilens/' | relative_url }}) (다리 속도 bias 를 상태로 추정),
> [LIJO]({{ '/study/slam/state-estimation/lijo/' | relative_url }}) (관절 속도 + 동적 가중치),
> [FAST-LIO2]({{ '/study/slam/state-estimation/fast-lio2/' | relative_url }})
