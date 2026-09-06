---
layout: study-chapter
title: "Chapter 4. 좌표계와 회전"
description: "3D SLAM 수식을 읽기 위한 최소한의 기하학."
importance: 4
category: SLAM
series: slam_history
permalink: /study/slam/history/04-coordinate-rotation/
---

> **목표:** 변환 방향을 명시하고 회전을 단순 덧셈으로 갱신하지 않는 이유를 이해한다.  
> **학습량:** 15분. 행렬-벡터 곱이 선행 지식이다.

## 1. 같은 점에도 좌표가 여러 개다

이 노트에서 $T_{AB}$는 **B 좌표를 A 좌표로 바꾸는 변환**이다. 센서가 관측한 점을 세계 좌표로 바꾸면:

$$
p_W=R_{WS}p_S+t_{WS}
$$

$R$은 회전, $t$는 이동이다. $t_{WS}$는 센서 원점의 세계 좌표다. 프레임을 연결할 때는 $T_{WC}=T_{WB}T_{BC}$처럼 중간 좌표가 맞아야 한다. 좌표 변환과 관측 모델은 [Course on SLAM](https://gisbi-kim.github.io/materials/study/soal17courseslam.pdf)의 기초 부분에 나온다.

## 2. 회전 행렬을 그냥 더하면?

회전 행렬은 $R^TR=I$, $\det R=1$을 만족해야 한다. 임의의 행렬을 더하면 이 조건을 잃을 수 있다. 3D 회전의 집합을 $SO(3)$라고 부른다.

작은 변화량 $\delta\theta$를 구한 뒤, 예를 들어 다음처럼 회전을 합성할 수 있다.

$$
R_{\text{new}}=R\operatorname{Exp}([\delta\theta]_\times)
$$

여기서는 오른쪽 perturbation을 사용했다. 왼쪽 갱신을 쓰는 문헌도 있으므로 Jacobian을 그대로 섞으면 안 된다. Quaternion도 단위 길이와 성분 순서 등 convention 확인이 필요하다. [Solà의 Quaternion kinematics](https://arxiv.org/abs/1711.02508)는 표현 간 관계와 perturbation을 상세히 다룬다.

## 3. 손계산으로 방향 확인하기

센서가 세계 원점에서 x 방향 2m에 있고 회전은 없다고 하자. 센서가 보는 점이 $p_S=(1,0,0)$이면 세계 좌표는 $(3,0,0)$이다.

반대로 세계 점을 센서 좌표로 바꾸려면:

$$
p_S=R_{WS}^{T}(p_W-t_{WS})
$$

이 예제에서 결과가 $(5,0,0)$으로 나오면 이동 부호나 변환 방향을 잘못 적용한 것이다. 복잡한 코드 전에 이런 단순 예제를 통과시키자.

## 면접형 확인 문제

### 문제 1 — 개념

로봇의 평행이동 궤적은 맞지만 point cloud가 robot origin 주위를 잘못된 방향으로 회전한다. 최적화 파라미터를 조정하기 전에 확인할 항목을 우선순위대로 설명하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

먼저 $T_{BS}$와 $T_{SB}$ 중 어느 방향의 extrinsic을 코드가 요구하는지 확인한다. 다음으로 active/passive rotation, quaternion 성분 순서 `(w,x,y,z)` 또는 `(x,y,z,w)`, degree/radian, 좌표축 handedness와 ROS optical frame 규약을 점검한다. Timestamp가 어긋나면 회전 운동 중 비슷한 현상이 생기므로 시간 동기화도 확인한다. 단위 변환과 단순한 알려진 pose 예제를 통과시킨 뒤 residual과 optimizer를 조사하는 순서가 효율적이다.

</details>

### 문제 2 — 수학

$T_{AB}=(R_{AB},t_{AB})$, $T_{BC}=(R_{BC},t_{BC})$이고 $p_A=R_{AB}p_B+t_{AB}$로 정의한다. $T_{AC}$의 회전과 이동을 유도하고, $T_{AB}^{-1}$을 구하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

$p_B=R_{BC}p_C+t_{BC}$를 첫 식에 대입하면

$$
p_A=R_{AB}R_{BC}p_C+R_{AB}t_{BC}+t_{AB}.
$$

따라서

$$
R_{AC}=R_{AB}R_{BC},\qquad
t_{AC}=R_{AB}t_{BC}+t_{AB}.
$$

역변환은 $p_B=R_{AB}^T(p_A-t_{AB})$이므로

$$
T_{AB}^{-1}=(R_{AB}^T,-R_{AB}^Tt_{AB}).
$$

이동 벡터를 단순히 빼는 것이 아니라 역회전까지 적용해야 한다.

</details>

## 원문 읽기

- Quaternion kinematics: §2의 회전 표현, §3의 convention, §4의 perturbation 중 필요한 부분. 로컬: `_resource/slam/foundations/sola2017-quaternion-eskf.pdf`.
- 첫 회독에서는 Jacobian 전체를 유도하지 않아도 된다.
