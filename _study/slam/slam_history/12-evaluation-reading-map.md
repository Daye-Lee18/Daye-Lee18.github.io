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

| 시기 / 대표 사례 | 이 노트에서 배운 질문 | 복습 |
|---|---|---|
| 초기 확률적 SLAM | 위치와 지도의 오차가 어떻게 연결되나? | 1~2장 |
| FastSLAM, 2002 | 경로 조건으로 문제를 나눌 수 있나? | 3장 |
| Graph/smoothing 계열 | 과거 상태와 큰 문제를 어떻게 갱신하나? | 6~7장 |
| KinectFusion, 2011 / ORB-SLAM, 2015 | 센서와 지도 표현이 설계를 어떻게 바꾸나? | 8~9장 |
| LIO-SAM, 2020 / FAST-LIO2, 2021 preprint | 빠른 관성 예측을 어떻게 보정하나? | 10장 |
| DROID-SLAM, 2021 및 neural map 계열 | 무엇을 학습하고 무엇을 최적화하나? | 11장 |

이는 앞 장의 원문들을 연결한 학습용 연대표이며, 각 계열의 최초 발명 목록이나 완전한 역사표가 아니다. 병렬로 발전한 기법들을 하나의 교체 순서로 읽지 말자.

## 3. 추천 글과 자료를 사용하는 순서

[Giseop Kim의 SLAM Back-end 공부자료 추천 글](https://gisbi-kim.github.io/post/slam-textbooks/)은 rotation, least squares, sparsity를 공부할 자료들을 연결한다. 이 노트에서는 긴 교재를 한 번에 읽지 않도록 다음처럼 나눠 사용한다.

1. 좌표와 회전이 헷갈리면 4장과 Solà의 회전 자료로 돌아간다.
2. Residual과 Jacobian이 헷갈리면 5장의 작은 ICP 문제를 푼다.
3. 큰 solver의 구조가 궁금하면 6~7장과 factor graph 교재를 읽는다.
4. 센서 파이프라인이 궁금하면 8~10장 뒤에 관심 시스템의 overview와 코드를 연다.

[SLAM Handbook 공식 페이지](https://asrl.utias.utoronto.ca/~tdb/slam/)는 더 넓은 주제를 찾아가는 색인으로 사용한다. 필요한 장을 골라 읽고, 판본과 공개 저장소의 갱신 시점을 확인한다.

## 4. 마무리 활동

관심 시스템 하나에 대해 다음 여섯 줄을 작성한다: 입력 센서, 추정 상태, 관측 residual, 지도 표현, 전역 보정 유무, 실패할 환경.

**확인 예:** IMU가 들어간다고 전역 위치가 자동으로 관측되는 것은 아니다. “빠른 예측”과 “누적 오차를 잡아 줄 기준”을 분리해서 설명할 수 있으면 LIO 학습을 시작할 준비가 됐다.

## 다음 학습

[LIO 목차]({{ '/study/slam/lio/' | relative_url }})에서 IMU 모델 → deskew → error-state → LIO-SAM/FAST-LIO2 순으로 이어간다. 현재 LIO 페이지는 학습 범위를 안내하는 목차다.
