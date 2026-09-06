---
layout: study-chapter
title: "Chapter 5. ICP와 점군 정합"
description: "두 관측을 맞추는 문제에서 반복 최적화를 이해하기."
importance: 5
category: SLAM
series: slam_history
permalink: /study/slam/history/05-registration-icp/
---

> **목표:** correspondence와 pose update를 구분한다.  
> **학습량:** 15분. Chapter 4의 좌표 변환을 사용한다.

## 1. 두 점군이 서로 어긋나 있다

다른 위치에서 얻은 점군을 같은 좌표계로 맞추려면 회전과 이동이 필요하다. Point-to-point ICP의 대표적인 목적함수는 다음과 같다.

$$
\min_{R,t}\sum_i\|Rp_i+t-q_{c(i)}\|^2
$$

$c(i)$는 source 점 $p_i$에 대응시킨 target 점의 번호다. 기본 흐름은 **현재 변환으로 대응점을 찾고 → 대응을 고정해 변환을 개선하고 → 다시 대응을 찾는 것**이다. [Giseop Kim의 SLAM Back-end 자료 안내](https://gisbi-kim.github.io/post/slam-textbooks/)가 추천하는 Grisetti의 *From Least-Squares to ICP*는 이 작은 문제에서 residual과 Jacobian을 연결한다.

## 2. 오차를 줄이는 것만으로 충분할까?

반복해서 같은 모양이 나오는 복도에서 옆 기둥을 대응점으로 잡아도 작은 오차를 만들 수 있다. 초기값, 겹치는 영역, 잘못된 점 제거가 중요한 이유다.

Point-to-plane은 점과 대응 평면 사이의 법선 방향 거리를 사용한다. 벽 하나만 보이면 벽을 따라 미끄러지는 움직임은 이 거리로 잘 구분하지 못한다. 측정이 어떤 방향을 제약하는지 확인해야 한다. 점군 정합의 발전은 [KRoC 정합 강연](https://drive.google.com/file/d/1fwaHF77iwTxcDVl1s1h8BcCZE0EvPmUC/view)으로 확장해서 읽는다.

## 3. 1차원으로 한 번 계산하기

대응이 이미 알려져 있고 회전이 없다고 하자. Source가 `[0, 1, 2]`, target이 `[2, 3, 4]`라면 residual은 $r_i=p_i+t-q_i$다. 제곱합을 최소화하는 이동은 $t=2$다.

그런데 마지막 target을 40으로 잘못 매칭하면 최적 이동은 $(2+2+38)/3=14$가 된다. 계산은 정확해도 대응이 틀리면 추정은 틀린다. Robust loss와 correspondence 검증은 서로 보완한다.

## 면접형 확인 문제

### 문제 1 — 개념

ICP가 낮은 residual로 수렴했지만 추정 pose가 틀렸다. 가능한 원인 세 가지와 각각을 확인할 진단 방법을 설명하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

첫째, 반복 구조에서 잘못된 local minimum에 수렴할 수 있으므로 초기값을 바꾸거나 global registration 결과와 비교한다. 둘째, 잘못된 correspondence와 outlier가 목적함수를 지배할 수 있으므로 대응 거리·법선 각도 분포와 inlier ratio를 확인한다. 셋째, 평면이나 긴 복도처럼 특정 운동 방향이 관측되지 않는 degeneracy일 수 있으므로 Hessian 또는 normal matrix의 eigenvalue와 condition number를 본다. 겹침이 적거나 시간 왜곡이 있는 경우도 raw scan과 deskew 결과를 시각화해 확인한다.

</details>

### 문제 2 — 수학

Point-to-plane ICP의 residual을 $r_i=n_i^T(Rp_i+t-q_i)$로 둔다. 현재 추정 근처에서 왼쪽 작은 회전 $R'\approx(I+[\delta\theta]_\times)R$과 이동 변화 $\delta t$를 적용할 때, $\delta\xi=[\delta\theta^T,\delta t^T]^T$에 대한 Jacobian 한 행을 구하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

$a_i=Rp_i$라고 두면

$$
R'p_i\approx a_i+[\delta\theta]_\times a_i
=a_i-[a_i]_\times\delta\theta.
$$

따라서

$$
r_i'\approx r_i+n_i^T\left(-[a_i]_\times\delta\theta+\delta t\right)
$$

이고 Jacobian은

$$
J_i=\begin{bmatrix}-n_i^T[a_i]_\times & n_i^T\end{bmatrix}.
$$

오른쪽 perturbation이나 다른 pose convention을 사용하면 회전 Jacobian의 형태가 달라진다. 유도에서 사용한 convention을 코드와 맞추는 것이 핵심이다.

</details>

## 원문 읽기

- KRoC 정합 강연: ICP에서 robust/global/learning 접근으로 연결되는 부분. 로컬: `_resource/slam/kroc2026/06-registration-hyungtae-lim.pdf`.
- Grisetti (2016)는 추천 글의 설명과 링크를 참고한다. 기존 직접 PDF 주소는 현재 404를 반환하므로 로컬 자료에는 포함하지 않았다.
