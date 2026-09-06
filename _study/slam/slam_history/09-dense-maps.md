---
layout: study-chapter
title: "Chapter 9. Sparse map에서 Dense map으로"
description: "지도는 무엇을 저장하고 어떤 질문에 답해야 할까?"
importance: 9
category: SLAM
series: slam_history
permalink: /study/slam/history/09-dense-maps/
---

> **목표:** sparse point, occupancy, surface 표현을 목적에 맞게 구분한다.  
> **학습량:** 10~15분. Chapter 8 이후 읽는다.

## 1. 위치를 찾기 좋은 지도와 표면을 보는 지도

벽 모서리의 특징점 몇 개만으로도 카메라 위치를 추정할 수 있다. 그러나 그 점들만으로 벽 전체 표면이나 빈 공간을 판단하기는 어렵다. 지도 표현은 사용할 작업의 요구에 맞춰 선택한다.

| 표현             | 저장하는 것                           | 먼저 물을 질문                    |
| ---------------- | ------------------------------------- | --------------------------------- |
| Sparse landmarks | 재관측 가능한 점의 위치               | 대응점을 다시 찾을 수 있나?       |
| Occupancy grid   | 셀의 점유 가능성                      | 빈 공간과 미관측 공간을 구분하나? |
| TSDF             | 표면 근처의 truncated signed distance | 표면을 어떻게 융합하나?           |
| Mesh / surfels   | 면 또는 국소 표면 요소                | 표면 갱신 비용은 얼마인가?        |

이 비교는 지도를 고를 때 사용할 학습용 체크리스트다. 각 표현이 제공하지 않는 정보를 추정 결과로 착각하지 않는 것이 중요하다.

## 2. Depth camera와 KinectFusion

[KinectFusion (2011)](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/)은 움직이는 depth camera로 표면을 밀집 재구성하는 대표적인 전환점이다. 카메라 tracking과 volumetric fusion을 연결해 새 depth를 기존 모델에 반영한다. 관측을 누적하면 잡음을 줄일 수 있지만, pose가 잘못되거나 물체가 움직이면 중복 표면과 흔적이 생길 수 있다.

[KRoC 3D World 강연](https://drive.google.com/file/d/1OTZjzUGls3fjSQed7LU-xzjS_78e7BIW/view)은 dense/volumetric SLAM을 feature 및 direct 방식과 연결해서 다룬다.

## 3. 메모리를 직접 계산하기

학습용 단순 grid를 가정하자. 10m 정육면체 공간을 10cm voxel로 나누면 $100^3=1,000,000$개다. 한 voxel에 8 byte를 쓰면 약 8MB다.

해상도를 5cm로 줄이면 $200^3=8,000,000$개, 약 64MB가 된다. 좌표당 두 배 정밀해졌지만 메모리는 여덟 배다. 이는 dense allocation만의 계산이며 index, buffer, sparse data structure 비용은 포함하지 않았다.

## 면접형 확인 문제

### 문제 1 — 개념

Occupancy map에서 `free`, `occupied`, `unknown`을 구분해야 하는 이유를 설명하라. Point cloud에서 점이 없는 공간을 모두 free로 표시하면 어떤 실패가 발생하는가?

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

Free는 센서 ray가 통과했다는 관측 증거가 있는 공간이고, occupied는 반사점 또는 표면이 관측된 공간이며, unknown은 아직 관측되지 않았거나 가려진 공간이다. 점이 없다는 이유만으로 unknown을 free로 바꾸면 센서 범위 밖, 물체 뒤쪽, 반사가 약한 표면을 안전한 공간으로 오판할 수 있다. Ray casting, sensor minimum/maximum range와 invalid return 모델을 사용해 free-space evidence를 명시적으로 갱신해야 한다.

</details>

### 문제 2 — 수학

한 occupancy cell의 prior 확률이 $p(m)=0.5$다. 서로 독립이라고 가정한 두 관측의 inverse sensor model이 각각 $p(m\mid z_1)=0.7$, $p(m\mid z_2)=0.8$을 준다. Log-odds update로 posterior occupancy 확률을 구하라.

<details class="study-answer" markdown="1">
<summary>답변 보기</summary>

Log-odds는 $l(p)=\log\frac{p}{1-p}$다. Prior $0.5$의 log-odds는 0이므로

$$
l=\log\frac{0.7}{0.3}+\log\frac{0.8}{0.2}
=\log\left(\frac{28}{3}\right)\approx2.234.
$$

이를 확률로 되돌리면

$$
p=\frac{1}{1+e^{-l}}=\frac{28}{31}\approx0.903.
$$

실제 연속 scan은 pose 오차와 중복 관측 때문에 완전히 독립이 아닐 수 있다. 같은 정보를 반복해 과도하게 확신하지 않도록 log-odds clamp와 sensor model 검증이 필요하다.

</details>

## 원문 읽기

- KinectFusion 원문: pipeline 그림과 volumetric integration 설명. 로컬: `_resource/slam/papers/kinectfusion2011.pdf`.
- KRoC 3D World: PDF 40~43쪽부터 읽는다.
