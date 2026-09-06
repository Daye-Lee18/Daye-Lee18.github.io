---
layout: study-chapter
title: "Chapter 11. 학습 기반 SLAM과 Spatial AI"
description: "학습은 파이프라인의 어느 부분을 바꾸는가?"
importance: 11
category: SLAM
series: slam_history
permalink: /study/slam/history/11-learning-spatial-ai/
---

> **목표:** learned feature, learned depth, neural map을 구분한다.  
> **학습량:** 10~15분. 최신 순위보다 구성 요소를 읽는 장이다.

## 1. “딥러닝 SLAM”을 한 종류로 묶지 않기

학습은 대응점 검출, descriptor, depth 추정, pose update, 지도 표현 등 서로 다른 위치에 들어갈 수 있다. [KRoC AI Visual SLAM 강연](https://drive.google.com/file/d/1-FZ207zXWqZiEudDnd5EEAaybWdIJzNe/view)은 learned features와 depth, neural SLAM을 구분해 보여준다. 모델 이름을 외우기 전에 입출력과 최적화 변수를 적어 보자.

[DROID-SLAM](https://arxiv.org/abs/2108.10869)은 recurrent update와 Dense Bundle Adjustment를 결합해 pose와 pixelwise depth를 갱신한다. 학습을 활용하면서도 기하학적 최적화 구조를 유지하는 사례다. 이 한 사례의 성능을 모든 학습 기반 접근이나 모든 환경으로 일반화하지 않는다.

## 2. 보기 좋은 지도와 위치 추정 성능

NeRF는 장면을 radiance field로 표현하고, 3D Gaussian Splatting은 Gaussian primitive를 이용해 렌더링한다. 이 표현을 지도에 활용할 수 있지만, 지도 표현 자체와 완전한 SLAM 시스템은 구분해야 한다. 카메라 pose가 이미 주어진 재구성과 pose도 함께 추정하는 SLAM은 입력 조건이 다르다. 표현의 기초는 [KRoC 3D Vision 강연](https://drive.google.com/file/d/1mL52klpHEYU6e-yZk3guMaJocLthSAA7/view)에서 읽는다.

## 3. 작은 논문 분석 활동

다음 표를 새 논문마다 한 줄씩 채운다.

| 항목          | 기록할 내용                                           |
| ------------- | ----------------------------------------------------- |
| 입력          | RGB만 받나, depth/IMU/pose도 받나?                    |
| 학습하는 부분 | correspondence, depth, update, map 중 무엇인가?       |
| 남은 기하학   | projection, pose optimization, loop 검증은 어디 있나? |
| 실행 조건     | GPU, 해상도, memory, 사전 학습 데이터는?              |
| 출력          | trajectory, surface, semantic label 중 무엇인가?      |

예를 들어 새 시점의 이미지가 선명해도 로봇 궤적이 미터 단위로 정확한지는 별도 실험이 필요하다. 사람에게 자연스러운 렌더링과 충돌 회피용 지도가 요구하는 정보도 다르다.

## 면접형 확인 문제

### 문제 1 — 개념

학습 기반 Visual SLAM이 학습 데이터에서는 기존 방법보다 좋지만 새로운 도시에서 크게 실패했다. 연구 면접에서 원인 분석과 추가 실험을 어떻게 제안하겠는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

먼저 학습 데이터와 새 도시 사이의 appearance, camera intrinsics, motion, weather, dynamic object 분포 차이를 확인한다. Learned feature, depth, pose update, map representation 중 어느 module에서 성능이 무너지는지 classical component로 교체하는 ablation을 수행한다. Geometry consistency, uncertainty calibration, catastrophic failure rate를 평균 ATE와 함께 측정한다. 여러 도시를 leave-one-domain-out으로 평가하고, 조도·blur·intrinsics를 통제한 실험으로 원인을 분리한다. Test-time adaptation을 쓴다면 ground-truth leakage, 계산량과 online stability도 보고해야 한다.

</details>

### 문제 2 — 수학·평가

Depth estimator가 모든 true depth $d_i$에 대해 $\hat d_i=1.2d_i$를 출력한다고 하자. Absolute relative error

$$
\mathrm{AbsRel}=\frac1N\sum_i\frac{|\hat d_i-d_i|}{d_i}
$$

를 구하라. 이 결과가 monocular SLAM의 pose 정확도를 충분히 설명하지 못하는 이유도 말하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

각 점에서

$$
\frac{|1.2d_i-d_i|}{d_i}=0.2
$$

이므로 $\mathrm{AbsRel}=0.2$, 즉 20%다. 그러나 이 값은 depth의 전역 scale bias만 요약하며 frame 사이의 temporal consistency, correspondence, camera pose, loop closure와 drift를 직접 측정하지 않는다. Monocular trajectory를 Sim(3)로 정렬하면 일정 scale bias가 평가에서 제거될 수도 있다. Depth metric과 trajectory metric, failure rate를 함께 봐야 한다.

</details>

## 원문 읽기

- KRoC AI Visual SLAM: PDF 31~42쪽. 로컬: `_resource/slam/kroc2026/07-ai-visual-slam-alex-lee.pdf`.
- DROID-SLAM: architecture 그림. 로컬: `_resource/slam/papers/droid-slam2021.pdf`.
- KRoC 3D Vision: PDF 59~63, 81~86쪽. 처음부터 모든 neural rendering 수식을 읽을 필요는 없다.
