---
layout: study-chapter
title: "Chapter 8. Visual SLAM과 Bundle Adjustment"
description: "카메라 관측에서 pose와 3D point를 함께 개선하기."
importance: 8
category: SLAM
series: slam_history
permalink: /study/slam/history/08-visual-slam/
---

> **목표:** reprojection error와 tracking/mapping의 역할을 이해한다.  
> **학습량:** 15분. Chapter 4와 Chapter 6을 먼저 읽는다.

## 1. 사진에는 깊이가 직접 보이지 않는다

한 픽셀은 카메라에서 나가는 시선 방향을 알려주지만 그 위의 거리를 바로 정하지는 못한다. 서로 다른 시점의 대응과 기하학을 사용해 3D 구조를 추정한다. 카메라 모델과 pose 추정의 연결은 [KRoC 3D Vision 강연](https://drive.google.com/file/d/1mL52klpHEYU6e-yZk3guMaJocLthSAA7/view)의 camera projection과 relative pose 부분을 참고한다.

## 2. Bundle Adjustment는 무엇을 맞추나?

추정한 세계 점 $P_j$를 camera pose로 변환한 뒤 영상에 투영해 관측 픽셀 $u_{ij}$와 비교한다.

$$
r_{ij}=u_{ij}-\pi(T_{C_iW}P_j)
$$

Bundle Adjustment(BA)는 이 reprojection residual을 사용해 camera와 point 변수를 함께 개선한다. Robust cost, gauge freedom, 희소성이 중요한 구현 요소다. BA는 SLAM 이전의 photogrammetry와 시각 재구성에도 뿌리를 둔다. [Triggs 등의 원문](https://lear.inrialpes.fr/people/triggs/pubs/Triggs-va99.pdf)은 이러한 배경과 수치 최적화를 다룬다.

## 3. 모든 프레임을 계속 최적화할까?

[ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956)은 tracking, local mapping, loop closing을 분리하며 ORB feature를 여러 작업에 활용하는 대표 사례다. 핵심 장면을 keyframe으로 유지하는 이유를 생각하면 계산량과 지도 관리가 연결된다.

영상의 photometric error를 사용하는 direct 방식도 있다. “feature/direct”는 관측 오차를 구성하는 방법의 구분이고, “sparse/dense”는 얼마나 많은 정보를 사용하는지와 관련된다. 같은 구분으로 취급하지 말자. [KRoC 3D World 강연](https://drive.google.com/file/d/1OTZjzUGls3fjSQed7LU-xzjS_78e7BIW/view)은 이 계열들을 비교한다.

## 4. Scale 사고 실험

순수 monocular 기하에서 장면과 카메라 이동을 함께 두 배로 늘려도 같은 영상 투영을 만들 수 있다. 알려진 길이, stereo baseline 등 추가 정보 없이 절대 크기를 정하기 어렵다는 뜻이다. 학습 depth를 쓰는 경우에는 데이터에서 얻은 prior와 일반화 조건을 별도로 살펴야 한다.

## 면접형 확인 문제

### 문제 1 — 개념

Monocular Visual SLAM의 reprojection error가 매우 작지만 trajectory scale이 틀릴 수 있는 이유를 observability 관점에서 설명하라. 어떤 정보가 scale을 정할 수 있는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

순수 monocular projection에서는 모든 3D point와 camera translation을 같은 비율 $s$로 늘려도 정규화 영상 좌표가 변하지 않는다. 따라서 영상 관측만으로 global metric scale은 관측 불가능한 gauge freedom이다. 알려진 stereo baseline, calibrated depth, 올바르게 모델링한 IMU와 중력·동역학 정보, wheel odometry, 알려진 물체 크기 또는 metric prior가 scale을 제공할 수 있다. 평가 때 Sim(3) 정렬을 사용하면 scale 오차가 제거되므로 SE(3) 정렬 결과와 구분해야 한다.

</details>

### 문제 2 — 수학

Pinhole camera에서 $u=f_xX/Z+c_x$, $v=f_yY/Z+c_y$다. 3D camera point $[X,Y,Z]^T$에 대한 projection Jacobian $\partial[u,v]/\partial[X,Y,Z]$를 구하라. $Z$가 0에 가까워질 때 어떤 수치 문제가 생기는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

Jacobian은

$$
J_\pi=
\begin{bmatrix}
f_x/Z & 0 & -f_xX/Z^2\\
0 & f_y/Z & -f_yY/Z^2
\end{bmatrix}.
$$

$Z$가 0에 가까우면 각 항이 매우 커져 작은 3D 변화가 큰 pixel 변화로 나타나며 선형화가 불안정해진다. $Z\le0$인 점은 camera 앞의 유효 관측이 아니므로 최적화에 넣기 전에 cheirality와 최소 depth를 검사해야 한다.

</details>

## 원문 읽기

- ORB-SLAM: system overview 그림과 세 thread 설명. 로컬: `_resource/slam/papers/orb-slam2015.pdf`.
- KRoC 3D Vision: PDF 31~38쪽. BA 교재는 Introduction과 cost function부터 읽는다.
