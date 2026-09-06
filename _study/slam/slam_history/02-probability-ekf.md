---
layout: study-chapter
title: "Chapter 2. 확률과 EKF-SLAM"
description: "한 점의 정답 대신 불확실성을 추적하는 방법."
importance: 2
category: SLAM
series: slam_history
permalink: /study/slam/history/02-probability-ekf/
---

> **목표:** 예측·보정과 로봇-지도 사이의 상관관계를 이해한다.  
> **학습량:** 15분. 평균과 분산의 의미를 알고 있으면 좋다.

## 1. 좌표 하나로는 부족하다

바퀴가 1m 굴렀다는 기록이 실제 1m 이동을 보장하지 않는다. 미끄러짐과 센서 오차가 있기 때문이다. 그래서 추정값과 함께 그 값을 얼마나 확신하는지 표현한다.

EKF-SLAM은 비선형 모델을 현재 추정 주변에서 선형화하며, 로봇 상태와 landmark를 함께 다루는 대표적인 filtering 접근이다. 다음 식은 문제의 구조를 보여준다.

$$
x_t=f(x_{t-1},u_t)+w_t, \qquad z_t=h(x_t,m_j)+v_t
$$

여기서 $u_t$는 이동 입력, $z_t$는 landmark $m_j$의 관측, $w_t,v_t$는 모델에 넣은 잡음이다. **예측**에서는 이동 모델을 적용하고, **보정**에서는 예상 관측과 실제 관측의 차이를 사용한다. 좌표계·모델의 기초는 [Solà의 Course on SLAM](https://upcommons.upc.edu/handle/2117/337287)에서 확인할 수 있다.

## 2. 지도 점들은 왜 서로 연결될까?

로봇이 자기 위치를 오른쪽으로 잘못 알고 두 기둥을 관측하면, 두 기둥 위치에도 공통 오차가 들어간다. 따라서 각 기둥을 독립적인 정답처럼 저장하면 안 된다.

공분산의 대각 성분은 각 변수의 분산을, 비대각 성분은 변수 사이의 상관관계를 담는다. 기존 landmark를 다시 관측하면 로봇뿐 아니라 다른 landmark 추정도 영향을 받을 수 있다. 이 연결은 유용하지만 큰 지도를 다룰 때 비용이 커진다. [FastSLAM 원논문의 Introduction](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf)은 기존 EKF 접근의 확장성 문제를 출발점으로 삼는다.

## 3. 1차원 보정 예제

지도상의 기준점이 알려져 있다고 단순화하자. 예측 위치가 2.0m, 분산이 0.25이고, 독립적인 위치 관측이 2.4m, 분산이 0.04라면:

$$
K=\frac{0.25}{0.25+0.04}\approx0.862,\qquad
\hat x=2.0+K(2.4-2.0)\approx2.345
$$

이 예제에서는 더 작은 분산을 가진 관측 쪽으로 많이 움직인다. 실제 EKF-SLAM에서는 관측이 위치 그 자체가 아니라 거리·각도일 수 있어 Jacobian과 상관관계를 포함해야 한다.

## 면접형 확인 문제

### 문제 1 — 개념

EKF-SLAM에서 로봇 pose와 landmark 사이의 cross-covariance를 모두 0으로 유지하면 어떤 문제가 생기는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

같은 불확실한 robot pose에서 관측된 landmark들은 공통 위치 오차를 공유한다. Cross-covariance를 0으로 만들면 이 상관관계를 잃어버리고, landmark 재관측이 robot pose와 다른 landmark에 전달해야 할 정보를 차단한다. 필터가 실제보다 독립적인 측정이 많다고 해석해 과도하게 확신하는 inconsistency가 생길 수 있다. 계산량을 줄이기 위한 근사는 가능하지만, 어떤 상관관계를 버렸으며 consistency를 어떻게 관리하는지 설명해야 한다.

</details>

### 문제 2 — 수학

1차원 상태의 prior가 $x\sim\mathcal N(2.0,0.25)$이고 측정 모델이 $z=x+v$, $v\sim\mathcal N(0,0.04)$이다. 실제 측정이 $z=2.4$일 때 Kalman gain, posterior mean, posterior variance를 구하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

측정 Jacobian은 $H=1$이므로

$$
K=\frac{0.25}{0.25+0.04}\approx0.8621
$$

이다. 따라서

$$
\hat x^+=2.0+0.8621(2.4-2.0)\approx2.3448,
$$

$$
P^+=(1-K)0.25\approx0.0345.
$$

측정 분산이 prior 분산보다 작아 평균은 측정 쪽으로 크게 이동하고 posterior 불확실성은 감소한다. 실제 EKF에서는 $H$가 현재 추정점에서 선형화된 Jacobian이므로 초기 추정과 선형화 오차도 함께 고려해야 한다.

</details>

## 원문 읽기

- Course on SLAM: motion/observation model 부분만 읽는다. 로컬: `_resource/slam/foundations/sola2017-course-on-slam.pdf`.
- FastSLAM (2002): Introduction의 EKF 한계 설명. 다음 장에서 조건부 독립으로 이어간다.
