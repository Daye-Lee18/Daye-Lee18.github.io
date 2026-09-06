---
layout: study-chapter
title: "Chapter 7. 희소성과 iSAM"
description: "변수가 많아도 모든 변수가 서로 연결되는 것은 아니다."
importance: 7
category: SLAM
series: slam_history
permalink: /study/slam/history/07-sparsity-isam/
---
> **목표:** sparsity, fill-in, incremental update의 의미를 구분한다.  
> **학습량:** 15분. Chapter 6 이후 읽는다.

## 1. 큰 행렬을 무조건 역행렬로 풀지 않는다

현재 추정 근처에서 residual을 선형화하면 $r(X+\delta)\approx r(X)+J\delta$가 된다. 가중치를 흡수한 Jacobian을 $A$, 우변을 $b$라고 쓰면 한 반복의 문제는 다음 형태다.

$$
\min_\delta\|A\delta-b\|^2
$$

하나의 관측은 대개 일부 변수에만 의존한다. 그래서 $A$에는 0인 블록이 많다. 실제 구현에서는 큰 역행렬을 직접 만드는 대신 QR이나 Cholesky 등의 분해로 선형계를 푼다.

## 2. 희소한 입력도 계산 중 채워질 수 있다

변수를 제거할 때 원래 없던 변수 사이의 연결이 생기는 현상이 fill-in이다. 제거 순서가 중간 행렬과 계산량을 바꾼다.

[Factor Graphs for Robot Perception](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html)은 이 구조를 그래프와 선형대수 양쪽에서 설명한다. Square Root SAM, iSAM, iSAM2로 이어지는 흐름에서는 희소 분해와 기존 계산의 재사용이 중요하다. 다만 큰 loop closure는 넓은 부분의 재계산을 요구할 수 있으므로 incremental이라는 말이 항상 일정 시간을 뜻하지 않는다.

## 3. 종이에 elimination 그려보기

세 pose A, B, C가 하나의 landmark L을 관측한다고 하자.

```text
A ── L ── B
     │
     C
```

L을 먼저 제거하면 A, B, C 사이에 관계가 남는다. 이제 A-B, B-C, A-C를 종이에 그려 보자. 제거 전보다 pose끼리 더 촘촘하게 연결된다. 이 그림이 fill-in의 직관이다. 좋은 ordering은 실제 그래프의 구조와 연결 정도에 따라 달라진다.

## 확인 질문

“측정 하나가 추가되었으니 기존 모든 계산을 항상 버려야 한다”는 말이 맞을까?

**확인:** 아니다. 변경된 부분과 영향을 받는 부분을 갱신하면서 재사용할 수 있다. 반대로 새 측정 하나가 멀리 떨어진 구간들을 연결한다면 영향 범위는 클 수 있다.

## 원문 읽기

- Dellaert & Kaess (2017): elimination/sparsity 설명을 먼저 읽고 iSAM 부분으로 넘어간다. 로컬: `_resource/slam/foundations/dellaert-kaess2017-factor-graphs.pdf`.
- 식 전체를 따라가기 전에 factor graph와 제거 후 그래프를 직접 그려본다.
