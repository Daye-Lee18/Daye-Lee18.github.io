---
layout: study-chapter
title: "Chapter 3. FastSLAM과 조건부 독립"
description: "큰 공동 추정 문제를 경로와 landmark 문제로 나누기."
importance: 3
category: SLAM
series: slam_history
permalink: /study/slam/history/03-fastslam/
---

> **목표:** FastSLAM의 핵심을 “경로가 주어졌을 때”라는 조건으로 설명한다.  
> **학습량:** 10~15분. Chapter 2 이후 읽는다.

## 1. 모든 점을 한꺼번에 추정해야 할까?

2002년의 [FastSLAM](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.html)은 robot path와 landmark의 확률 구조를 이용한다. 정적인 landmark와 조건부 독립 관측 등의 모델 가정 아래, **경로가 주어지면** landmark 추정들을 분리할 수 있다. 실제 경로는 모르므로 여러 경로 가설을 particle로 유지한다.

$$
p(x_{0:t},m\mid z_{1:t},u_{1:t})
=p(x_{0:t}\mid z_{1:t},u_{1:t})
\prod_j p(m_j\mid x_{0:t},z_{1:t})
$$

이 표기는 data association이 주어졌다는 조건을 생략한 학습용 형태다. 서로 다른 landmark의 관측 잡음이 독립이라는 가정도 필요하다.

## 2. Particle은 지도 위의 점 하나가 아니다

각 particle은 다른 경로 가설과 그 가설에 조건화된 landmark 추정들을 가진다. 관측이 들어오면 경로를 예측하고, 관련 landmark를 갱신하고, 관측을 잘 설명하는 가설에 큰 가중치를 준다. Resampling은 그 가중치를 반영해 가설을 다시 선택하는 단계다.

```text
가설 A: 복도에서 일찍 회전했다 → A에 맞춘 지도
가설 B: 복도에서 늦게 회전했다 → B에 맞춘 지도
                           ↓ 새로운 관측
                  더 잘 설명하는 가설에 무게
```

지도들이 아무 조건 없이 독립적인 것은 아니다. 각 가설 안에서 경로를 조건으로 두었기 때문에 분해가 가능하다.

## 3. 작은 사고 실험

같이 생긴 문이 두 개 있다. 로봇은 어느 문 앞에 있는지 확신하지 못한다. 평균 위치 하나만 그리면 두 문의 중간을 가리킬 수 있다. 두 경로 가설을 유지하면 실제 가능한 두 경우를 표현할 수 있다.

하지만 다음 관측도 두 문을 구분하지 못한다면, 가설을 여러 개 유지한다고 문제가 저절로 해결되지는 않는다. 가설 수와 계산량, resampling 이후의 다양성도 고민해야 한다.

## 면접형 확인 문제

### 문제 1 — 개념

FastSLAM에서 landmark들이 독립이라고 말할 때 정확히 어떤 조건이 붙는가? 이 조건을 빼고 “지도 점은 서로 독립이다”라고 설명하면 왜 틀리는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

정적인 landmark, 조건부 독립 관측 잡음, 알려진 data association 등의 모델 가정 아래 **robot path가 주어졌을 때** landmark posterior들이 조건부로 분리된다. 경로가 불확실하면 같은 경로 오차가 여러 landmark에 공통으로 영향을 주므로 marginal하게는 서로 의존한다. FastSLAM은 경로 가설을 particle로 표현하고, 각 particle 안에서 landmark별 추정기를 유지해 이 구조를 이용한다.

</details>

### 문제 2 — 수학·추론

Particle 세 개의 정규화 전 가중치가 $[0.1,0.3,0.6]$이다. Effective sample size를 $N_{eff}=1/\sum_i w_i^2$로 정의할 때 값을 구하고, resampling 기준이 $N/2$라면 resampling 여부를 판단하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

가중치는 이미 합이 1이므로

$$
N_{eff}=\frac{1}{0.1^2+0.3^2+0.6^2}
=\frac{1}{0.46}\approx2.17
$$

이다. $N=3$이고 기준은 $1.5$이므로 이 규칙에서는 resampling하지 않는다. 다만 $N_{eff}$ 하나만으로 가설의 기하학적 다양성이 충분한지는 알 수 없다. 너무 잦은 resampling은 particle impoverishment를 만들고, 너무 늦으면 낮은 가중치 particle 계산을 계속하게 된다.

</details>

## 원문 읽기

- Montemerlo et al. (2002), FastSLAM: Figure 1과 posterior 분해식 중심으로 읽는다. [원문 PDF](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf).
- 다음 장부터는 필터와 최적화 양쪽에 필요한 기하학을 잠시 정리한다.
