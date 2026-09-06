---
layout: study-chapter
title: "Chapter 10. IMU 융합에서 LIO까지"
description: "빠른 관성 예측과 외부 관측이 서로 필요한 이유."
importance: 10
category: SLAM
series: slam_history
permalink: /study/slam/history/10-inertial-lio/
---

> **목표:** IMU drift, deskew, filtering/smoothing의 관계를 설명한다.  
> **학습량:** 15분. 이 장은 다음 LIO 파트의 입구다.

## 1. IMU가 있으면 왜 도움이 될까?

IMU는 고속으로 각속도와 specific force를 제공한다. Accelerometer 값은 세계 좌표의 위치 가속도 그 자체가 아니다. 회전, 중력, bias를 고려해야 한다. 적분 과정에서 작은 오차가 누적되기 때문에 외부 센서 관측으로 보정하는 aided navigation이 필요하다. [KRoC IMU 강연](https://drive.google.com/file/d/1byqGAKCCsnv8rZbko4RBG9KiQQD_4h8x/view)은 INS의 문제와 보조 관측의 역할을 설명한다.

## 2. 한 scan도 한 순간에 찍힌 것이 아니다

회전식 LiDAR의 점들은 서로 다른 시각에 측정된다. 이동 중의 점군을 한 시각의 관측처럼 겹치면 벽이 휘거나 이중으로 보일 수 있다. Deskew는 측정 시점 사이의 움직임을 반영해 점들을 기준 시각으로 옮기는 과정이다. Timestamp와 extrinsic이 잘못되면 센서 수를 늘려도 해결되지 않는다.

## 3. 대표 시스템의 설계를 비교하기

| 시스템                    | 대표적인 추정 방식                        | 읽으면서 찾을 요소                                  |
| ------------------------- | ----------------------------------------- | --------------------------------------------------- |
| LIO-SAM (2020)            | Factor graph를 활용한 tightly coupled LIO | IMU preintegration, LiDAR odometry, loop/GPS factor |
| FAST-LIO2 (2021 preprint) | Iterated Kalman filter 기반 direct LIO    | 원시 점의 scan-to-map 정합, incremental map 관리    |

[LIO-SAM](https://arxiv.org/abs/2007.00258)은 IMU 정보를 deskew와 초기 추정에 사용하고 factor graph로 측정을 결합한다. [FAST-LIO2](https://arxiv.org/abs/2107.06829)는 직접적인 점군 정합과 효율적인 map 자료구조를 결합한다. 이 비교는 정확도 순위가 아니라 설계 차이다. FAST-LIO2의 local mapping만 보고 전역 loop closure가 자동 포함된다고 가정하면 안 된다.

## 4. 손계산 예제

자세 오차와 다른 잡음을 무시하고 가속도 bias가 일정하게 $0.01m/s^2$라고 하자. 10초 적분하면 위치 오차 항은 $\frac12bt^2=0.5m$다. 작은 bias도 시간이 지나면 무시하기 어렵다.

## 면접형 확인 문제

### 문제 1 — 개념

Filtering과 fixed-lag smoothing을 지연, 계산량, 재선형화, 과거 상태 수정 관점에서 비교하라. LIO 시스템이 두 접근 중 하나만 써야 하는지도 설명하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

Filtering은 보통 현재 상태와 covariance를 재귀적으로 갱신해 낮은 지연과 bounded state size를 얻기 쉽지만, marginalize한 과거 상태를 새 정보로 직접 다시 최적화하기 어렵다. Fixed-lag smoothing은 일정 시간 창의 여러 상태를 함께 유지해 재선형화와 delayed measurement 처리가 가능하지만 계산량과 지연이 커진다. 실전 시스템은 고속 propagation과 local odometry에 filter를 쓰고, keyframe graph 또는 loop closure에 smoothing을 쓰는 식으로 결합할 수 있다. 비교 축은 신구 방식이 아니라 latency, state horizon, nonlinear model과 전역 보정 요구다.

</details>

### 문제 2 — 수학

자세는 정확하고 초기 속도 오차는 0이라고 하자. Body x축 accelerometer bias가 $b_a=0.02m/s^2$로 일정하다. 20초 동안 보정 없이 적분했을 때 속도 오차와 위치 오차를 구하라. 또한 roll 오차 $1^\circ$가 있을 때 중력 $9.81m/s^2$가 수평축에 만드는 가속도 오차를 근사하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

일정 bias만 고려하면

$$
\delta v=b_at=0.02\times20=0.4m/s,
$$

$$
\delta p=\frac12b_at^2
=\frac12\times0.02\times20^2=4m.
$$

$1^\circ\approx0.01745rad$이므로 작은 각도 근사에서 중력 투영 오차는

$$
g\sin(1^\circ)\approx9.81\times0.01745\approx0.171m/s^2
$$

이다. 이는 위 bias보다 훨씬 크다. IMU 적분에서 자세 오차가 위치 오차로 빠르게 커지는 이유이며, 중력 정렬과 자세 보정이 중요하다.

</details>

## 원문 읽기

- KRoC IMU: PDF 13~24쪽. 로컬: `_resource/slam/kroc2026/05-imu-giseop-kim.pdf`.
- 두 LIO 논문은 abstract와 system overview만 읽는다. 세부 유도와 코드 분석은 LIO 파트에서 이어갈 주제다.
