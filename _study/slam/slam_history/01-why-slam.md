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

| 질문 | 이 노트에서 만날 답 |
|---|---|
| 오차를 어떻게 표현할까? | 확률 분포, 공분산, filtering |
| 큰 지도에서 계산을 어떻게 줄일까? | 조건부 독립, 희소 최적화 |
| 어떤 센서와 지도를 사용할까? | 카메라, LiDAR, IMU, 밀집 지도 |
| 실제 환경에서 계속 동작할까? | 재방문, 강건성, 학습, 평가 |

이 순서는 학습용이다. 실제 연구는 서로 겹쳐 발전했다. 새 방법이 나와도 이전 접근이 사라지는 것은 아니다.

## 4. 손으로 생각하기

정사각형 복도를 한 바퀴 돈 로봇이 출발점에서 0.8m 떨어진 곳에 도착했다고 추정했다. 마지막 위치만 출발점으로 강제로 옮기면 무엇이 남을까?

**확인:** 중간 궤적과 벽의 위치가 여전히 어긋날 수 있다. 전체 궤적을 고칠 때도 각 관측의 불확실성과 제약을 함께 고려해야 한다.

## 원문 읽기

- KRoC History 슬라이드: `What is SLAM?`, `History of SLAM` 부분. 로컬: `_resource/slam/kroc2026/01-history-ayoung-kim.pdf`.
- Cadena et al. (2016): Introduction만 먼저 읽는다. 로컬: `_resource/slam/papers/cadena2016-slam-survey.pdf`.
