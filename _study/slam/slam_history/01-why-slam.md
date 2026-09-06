---
layout: study-chapter
title: "Chapter 1. SLAM은 왜 필요한가?"
description: "위치와 지도의 관계, odometry와 SLAM, 발전의 큰 흐름."
importance: 1
category: SLAM
series: slam_history
permalink: /study/slam/history/01-why-slam/
---

> **목표:** 위치와 지도를 함께 추정하는 이유를 설명한다.  
> **학습량:** 본문 10분 + 확인 문제 5분. 선행 지식 없이 시작한다.

## 1. 지도가 있어야 위치를 알 수 있는데

처음 들어간 건물에서 로봇이 벽까지의 거리를 측정했다고 하자. 이 측정만으로 건물 전체에서 자신의 위치를 알 수는 없다. 반대로 벽을 지도에 그리려면 측정 당시 로봇이 어디에 있었는지 알아야 한다.

SLAM은 이 두 미지수, **로봇의 위치·자세와 주변 지도**를 관측으로 함께 추정한다. 지도는 꼭 사람이 보는 그림일 필요는 없다. 재관측할 특징점의 좌표도 지도다. KRoC의 [History 강연](https://drive.google.com/file/d/1tmWxcQFD0lGZPO3L6wjxT1k6am4EXyMK/view)은 위치 추정과 지도의 요구사항을 구분하며 이 문제를 소개한다.

## 2. Odometry와 loop closure

Odometry는 연속 관측에서 이동을 추정한다. 작은 오차가 쌓이면 출발점에 돌아와도 추정 궤적의 끝이 시작점과 어긋날 수 있다. Loop closure는 과거 장소의 재방문을 확인해 먼 시점 사이에 제약을 추가한다. 장소가 비슷해 보이는 것과 실제 같은 장소임을 기하학적으로 검증하는 것은 별개다.

```text
연속 관측 → 이동 추정 → 누적 궤적
                         ↑
과거 장소 재인식 → 기하 검증 → 재방문 제약
```

Loop closure를 항상 성공해야만 SLAM인 것은 아니다. 지역 지도만 유지하는 시스템도 있다. 논문을 읽을 때는 이름보다 **추정하는 상태와 지도의 범위**를 확인하자. [Cadena 등의 SLAM survey](https://arxiv.org/abs/1606.05830)는 문제 정의와 전체 시스템의 구성, 남은 과제를 정리한다.

## 3. 역사를 읽는 네 가지 질문

| 질문                              | 이 노트에서 만날 답           |
| --------------------------------- | ----------------------------- |
| 오차를 어떻게 표현할까?           | 확률 분포, 공분산, filtering  |
| 큰 지도에서 계산을 어떻게 줄일까? | 조건부 독립, 희소 최적화      |
| 어떤 센서와 지도를 사용할까?      | 카메라, LiDAR, IMU, 밀집 지도 |
| 실제 환경에서 계속 동작할까?      | 재방문, 강건성, 학습, 평가    |

이 순서는 학습용이다. 실제 연구는 서로 겹쳐 발전했다. 새 방법이 나와도 이전 접근이 사라지는 것은 아니다.

## 4. 면접형 확인 문제

### 문제 1 — 개념

면접관이 “Odometry에 loop closure만 붙이면 항상 SLAM이라고 부를 수 있나요?”라고 물었다. 시스템 구성 요소와 추정 대상의 관점에서 답하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

Loop closure의 존재만으로 판단하기는 어렵다. 먼저 시스템이 로봇 상태와 환경 표현을 어떤 범위에서 추정하는지 확인해야 한다. Odometry는 주로 연속 시점 사이의 상대 운동을 누적하고, SLAM은 관측과 상태·지도 사이의 일관성을 함께 다룬다. 지역 지도만 유지하거나 loop closure가 없는 SLAM도 가능하다. 반대로 과거 pose만 보정하고 재사용 가능한 환경 표현을 만들지 않는 시스템은 pose-graph localization 또는 trajectory optimization에 가까울 수 있다. 따라서 입력 센서, 상태 변수, map representation, 재방문 제약이 실제로 갱신하는 변수를 설명한 뒤 용어를 결정해야 한다.

</details>

### 문제 2 — 수학·추론

정사각형 경로의 네 변을 각각 10m로 추정했는데 각 변의 이동량에 독립적인 표준편차 $0.1m$가 있다고 하자. 단순화를 위해 회전 오차를 무시할 때, 한 축에서 두 번의 반대 방향 이동이 상쇄된 뒤 남는 위치 오차의 표준편차는 얼마인가? 또한 마지막 pose만 원점으로 옮기는 방식의 한계를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

한 축에서 독립인 두 이동 오차의 분산은 더해지므로

$$
\sigma_x=\sqrt{0.1^2+0.1^2}\approx0.141m
$$

이다. 다른 축도 같은 가정이면 동일하다. 실제로는 회전 오차가 이후 이동 방향에 영향을 주므로 오차가 독립적인 단순 합보다 커지고 축 사이 상관관계도 생긴다. 마지막 pose만 원점으로 옮기면 loop의 끝점만 맞을 뿐, 중간 궤적과 지도에 분배된 오차는 남는다. 전체 pose와 관측 제약의 불확실성을 사용해 궤적 전체를 조정해야 한다.

</details>

## 원문 읽기

- KRoC History 슬라이드: `What is SLAM?`, `History of SLAM` 부분. 로컬: `_resource/slam/kroc2026/01-history-ayoung-kim.pdf`.
- Cadena et al. (2016): Introduction만 먼저 읽는다. 로컬: `_resource/slam/papers/cadena2016-slam-survey.pdf`.
