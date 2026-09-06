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

## 확인 질문

로봇 위치는 맞는데 point cloud가 로봇을 중심으로 회전해 보인다면 무엇을 먼저 확인할까?

**확인:** sensor-to-body extrinsic의 방향, 회전 단위, quaternion 성분 순서와 active/passive convention을 확인한다. 최적화 파라미터를 바꾸기 전에 좌표 정의가 일치하는지 본다.

## 원문 읽기

- Quaternion kinematics: §2의 회전 표현, §3의 convention, §4의 perturbation 중 필요한 부분. 로컬: `_resource/slam/foundations/sola2017-quaternion-eskf.pdf`.
- 첫 회독에서는 Jacobian 전체를 유도하지 않아도 된다.
