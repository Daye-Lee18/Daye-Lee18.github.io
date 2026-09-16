---
layout: page
title: SLAM
description: 설명을 읽고 문제를 풀며, SLAM의 큰 그림에서 실제 LiDAR–inertial 시스템까지 이어지는 입문 코스.
permalink: /study/slam/
topic_index: true
---

SLAM을 처음 공부한다면 논문 이름부터 외우지 않아도 된다. 먼저 **왜 위치와 지도를 함께 추정하는지**, 센서 측정이 어떻게 **상태 보정**으로 이어지는지, 누적 오차를 왜 **loop closure**로 다시 고치는지만 잡으면 된다.

이 페이지는 기존 자료를 세 단계로 묶은 학습 지도다. 각 장은 **설명 → 작은 수치 예제 → 확인 문제 → 원문 읽기** 순서로 되어 있다. 문제는 시험이 아니라, 설명을 정말 이해했는지 바로 확인하는 장치다.

## 먼저: 90분 빠른 입문 코스

처음에는 아래 일곱 장만 이 순서대로 읽는다. 각 장의 수식이 모두 익숙해질 때까지 멈추기보다, 질문에 말로 답할 수 있으면 다음 장으로 넘어간다.

<div class="table-responsive" markdown="1">

| 순서 | 읽을 내용 | 이 단계에서 답할 수 있어야 하는 질문 |
| :--: | --- | --- |
| 1 | [왜 SLAM이 필요한가]({{ '/study/slam/history/01-why-slam/' | relative_url }}) | Odometry와 SLAM은 무엇이 다르고, 오차는 왜 누적되는가? |
| 2 | [좌표계와 회전]({{ '/study/slam/history/04-coordinate-rotation/' | relative_url }}) | 같은 점의 좌표가 센서·로봇·월드 좌표계에서 왜 다른가? |
| 3 | [ICP와 point-cloud registration]({{ '/study/slam/history/05-registration-icp/' | relative_url }}) | 두 번의 관측을 맞추어 상대 움직임을 어떻게 구하는가? |
| 4 | [확률과 EKF-SLAM]({{ '/study/slam/history/02-probability-ekf/' | relative_url }}) | prediction과 correction은 무엇이며 covariance는 왜 필요한가? |
| 5 | [Graph SLAM과 loop closure]({{ '/study/slam/history/06-graph-slam/' | relative_url }}) | 재방문 제약 하나가 과거 trajectory 전체를 어떻게 고치는가? |
| 6 | [IMU fusion에서 LIO까지]({{ '/study/slam/history/10-inertial-lio/' | relative_url }}) | 빠르지만 drift하는 IMU와 LiDAR가 왜 서로 필요한가? |
| 7 | [평가와 다음 학습 지도]({{ '/study/slam/history/12-evaluation-reading-map/' | relative_url }}) | ATE·RPE가 각각 무엇을 측정하며 어떤 논문을 다음에 읽어야 하는가? |

</div>

> **빠른 통과 기준:** 각 장의 확인 문제를 먼저 풀고, 답을 펼쳐 자신의 설명과 비교한다. 계산이 막혀도 입력·출력과 오차가 보정되는 방향을 말로 설명할 수 있으면 1회독은 충분하다.

## 다음: 관심에 따라 가지를 고른다

빠른 입문 코스를 마친 뒤에는 모든 글을 순서대로 읽을 필요가 없다.

- **필터와 확률을 더 이해하고 싶다:** [FastSLAM과 conditional independence]({{ '/study/slam/history/03-fastslam/' | relative_url }})
- **큰 최적화 문제가 어떻게 빨라지는지 궁금하다:** [Sparsity와 iSAM]({{ '/study/slam/history/07-sparsity-isam/' | relative_url }})
- **카메라 SLAM을 보고 싶다:** [Visual SLAM과 Bundle Adjustment]({{ '/study/slam/history/08-visual-slam/' | relative_url }})
- **지도 표현을 비교하고 싶다:** [Sparse map에서 dense map으로]({{ '/study/slam/history/09-dense-maps/' | relative_url }})
- **학습 기반 방법이 바꾸는 부분을 알고 싶다:** [Learning-based SLAM과 Spatial AI]({{ '/study/slam/history/11-learning-spatial-ai/' | relative_url }})

전체 역사 코스와 원문 자료는 [SLAM History 목차]({{ '/study/slam/history/' | relative_url }})에서 볼 수 있다.

## 마지막: 실제 시스템과 프로젝트에 연결한다

<div class="study-section-list">
  <a class="study-section-card" href="{{ '/study/slam/state-estimation/' | relative_url }}">
    <span class="study-section-kicker">Step 2 · paper-to-system</span>
    <strong>LIO &amp; State Estimation</strong>
    <span>FAST-LIO2를 기준점으로 삼아 LOAM, LIO-SAM, Point-LIO, FAST-LIVO2, VILENS 등의 센서·상태·잔차·back-end를 비교한다.</span>
  </a>
  <a class="study-section-card" href="{{ '/study/slam/leg-fusion/' | relative_url }}">
    <span class="study-section-kicker">Step 3 · worked project</span>
    <strong>Leg-Velocity Fusion in FAST-LIO2</strong>
    <span>다리 운동학 속도를 FAST-LIO2 필터의 관측으로 추가하고, 좌표 변환·Kalman update·chi-square gate를 직접 계산한다.</span>
  </a>
</div>

논문 리뷰부터 시작하면 알고리즘 이름은 늘지만 서로의 차이가 잘 보이지 않는다. 먼저 빠른 입문 코스로 공통 언어를 만든 뒤, 현재 시스템인 FAST-LIO2를 기준으로 다른 논문을 비교하고, 마지막에 leg-velocity fusion 계산으로 연결하는 것이 이 자료의 기본 경로다.

## 전체 구조

```text
큰 그림과 공통 언어
  └─ 빠른 입문 7장
       ├─ 필요한 이론만 선택해서 보충
       └─ FAST-LIO2를 기준으로 논문 비교
             └─ Vision60 leg-velocity fusion 계산과 구현
```

다운로드한 공개 강의 자료와 논문은 `_resource/slam`에 보관했고, 출처와 원문 링크는 [resource index](https://github.com/Daye-Lee18/Daye-Lee18.github.io/tree/main/_resource/slam)에 기록했다.
