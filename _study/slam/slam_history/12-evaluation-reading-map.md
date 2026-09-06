---
layout: study-chapter
title: "Chapter 12. 평가와 다음 학습 경로"
description: "역사를 성능표와 코드 읽기로 연결하는 방법."
importance: 12
category: SLAM
series: slam_history
permalink: /study/slam/history/12-evaluation-reading-map/
---

> **목표:** SLAM 결과를 비교하기 전에 실험 조건을 맞춘다.  
> **학습량:** 15분. History 마지막 장이다.

## 1. 같은 숫자가 같은 실험을 뜻하지 않는다

Trajectory 오차를 비교할 때는 timestamp 대응, 좌표 정렬, scale 보정 여부, 실패한 구간의 처리부터 확인한다. 순수 monocular 추정에 scale 정렬을 허용한 결과와 metric sensor 추정 결과를 아무 설명 없이 비교하면 의미가 달라진다.

[KITTI odometry 평가](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)는 여러 길이의 부분 궤적에서 translation/rotation 오차를 평가한다. 단일 전체 궤적 오차 숫자와 같은 지표로 취급하지 않는다. 정확도와 함께 latency, memory, 실패율도 기록하면 실제 로봇에 적합한지 판단하기 쉽다.

## 2. 연대표는 위치를 잡는 도구

| 시기 / 대표 사례                         | 이 노트에서 배운 질문                    | 복습  |
| ---------------------------------------- | ---------------------------------------- | ----- |
| 초기 확률적 SLAM                         | 위치와 지도의 오차가 어떻게 연결되나?    | 1~2장 |
| FastSLAM, 2002                           | 경로 조건으로 문제를 나눌 수 있나?       | 3장   |
| Graph/smoothing 계열                     | 과거 상태와 큰 문제를 어떻게 갱신하나?   | 6~7장 |
| KinectFusion, 2011 / ORB-SLAM, 2015      | 센서와 지도 표현이 설계를 어떻게 바꾸나? | 8~9장 |
| LIO-SAM, 2020 / FAST-LIO2, 2021 preprint | 빠른 관성 예측을 어떻게 보정하나?        | 10장  |
| DROID-SLAM, 2021 및 neural map 계열      | 무엇을 학습하고 무엇을 최적화하나?       | 11장  |

이는 앞 장의 원문들을 연결한 학습용 연대표이며, 각 계열의 최초 발명 목록이나 완전한 역사표가 아니다. 병렬로 발전한 기법들을 하나의 교체 순서로 읽지 말자.

## 3. 추천 글과 자료를 사용하는 순서

[Giseop Kim의 SLAM Back-end 공부자료 추천 글](https://gisbi-kim.github.io/post/slam-textbooks/)은 rotation, least squares, sparsity를 공부할 자료들을 연결한다. 이 노트에서는 긴 교재를 한 번에 읽지 않도록 다음처럼 나눠 사용한다.

1. 좌표와 회전이 헷갈리면 4장과 Solà의 회전 자료로 돌아간다.
2. Residual과 Jacobian이 헷갈리면 5장의 작은 ICP 문제를 푼다.
3. 큰 solver의 구조가 궁금하면 6~7장과 factor graph 교재를 읽는다.
4. 센서 파이프라인이 궁금하면 8~10장 뒤에 관심 시스템의 overview와 코드를 연다.

[SLAM Handbook 공식 페이지](https://asrl.utias.utoronto.ca/~tdb/slam/)는 더 넓은 주제를 찾아가는 색인으로 사용한다. 필요한 장을 골라 읽고, 판본과 공개 저장소의 갱신 시점을 확인한다.

## 4. 면접형 확인 문제

### 문제 1 — 개념

두 SLAM 논문 중 A는 ATE가 더 작고 B는 RPE와 실패율이 더 작다. 회사의 장시간 자율주행 로봇에 적용할 방법을 고르라는 질문에 어떻게 답하겠는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

숫자 하나로 선택하지 않고 운영 조건부터 맞춘다. ATE는 전역 궤적 일치도를, RPE는 일정 구간의 local drift를 주로 반영하며 정렬 방식에도 민감하다. 장시간 로봇에서는 catastrophic failure, relocalization 시간, loop closure 오검출, memory 증가, latency tail과 recovery behavior가 중요하다. 동일 센서·연산 장치·데이터 구간과 같은 SE(3)/Sim(3) 정렬 조건에서 재평가하고, 실패 비용이 큰 제품이라면 평균 정확도보다 failure rate와 복구 가능성에 높은 가중치를 둘 수 있다. 최종 선택은 실제 운용 분포에서의 요구사항과 위험 허용치로 결정한다.

</details>

### 문제 2 — 수학·평가

2D trajectory의 추정점이 $(0,0),(1.1,0),(2.2,0)$이고 ground truth가 $(0,0),(1,0),(2,0)$이다. 시작점을 맞춘 상태에서 translation ATE RMSE와, 두 연속 구간의 translation RPE RMSE를 구하라. 두 지표가 무엇을 보여주는지도 설명하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

각 pose의 위치 오차 크기는 $0,0.1,0.2$이므로

$$
\mathrm{ATE}_{RMSE}
=\sqrt{\frac{0^2+0.1^2+0.2^2}{3}}
\approx0.129m.
$$

각 1m ground-truth 구간에 대해 추정 이동은 1.1m이므로 두 구간의 상대 translation 오차는 모두 0.1m다.

$$
\mathrm{RPE}_{RMSE}=\sqrt{\frac{0.1^2+0.1^2}{2}}=0.1m.
$$

RPE는 매 구간의 10cm drift를 보여주고, ATE는 그 drift가 누적되어 마지막 pose에서 20cm가 된 결과까지 반영한다. 실제 평가는 회전, timestamp association, trajectory alignment와 구간 길이 정의를 명시해야 한다.

</details>

## 다음 학습

[LIO 목차]({{ '/study/slam/lio/' | relative_url }})에서 IMU 모델 → deskew → error-state → LIO-SAM/FAST-LIO2 순으로 이어간다. 현재 LIO 페이지는 학습 범위를 안내하는 목차다.
