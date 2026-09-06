---
layout: study-chapter
title: "Chapter 6. Graph SLAM과 loop closure"
description: "과거 궤적을 다시 고칠 수 있는 제약 기반 표현."
importance: 6
category: SLAM
series: slam_history
permalink: /study/slam/history/06-graph-slam/
---
> **목표:** front-end, back-end, factor의 역할을 나눈다.  
> **학습량:** 15분. Chapter 5의 residual 개념을 사용한다.

## 1. 한 번 지나온 궤적도 수정할 수 있다

Graph SLAM에서는 pose나 landmark를 변수로, 관측 관계를 제약으로 표현한다. Front-end는 대응과 상대 운동 등의 측정을 만들고, back-end는 그 측정들이 가능한 한 일관되도록 상태를 조정한다. Factor graph는 변수와 factor를 별도 노드로 나타낸다.

가우시안 잡음을 가정한 대표적인 MAP 추정은 다음 가중 최소제곱 문제로 연결된다.

$$
X^*=\arg\min_X\sum_k r_k(X)^T\Omega_k r_k(X)
$$

$\Omega_k$는 정보 행렬이며 공분산의 역행렬에 해당한다. 수식의 상세한 확률적 연결은 [KRoC Back-end 강연](https://drive.google.com/file/d/1FGnya__7ZQYsgE7CRhRjggeU2fIQ3izH/view)을 참고한다.

## 2. 재방문 제약을 넣으면

```text
x0 ── x1 ── x2 ── x3
└─────────────────┘
      재방문 제약
```

연속 이동 측정만 있던 그래프에 $x_0$와 $x_3$ 사이의 관계가 추가되면 중간 pose들도 바뀔 수 있다. 단순히 마지막 점만 붙이는 작업이 아니다. 절대 기준이 없는 상대 측정 그래프는 전체를 함께 이동·회전해도 같은 residual을 갖는다. 기준 pose를 고정하거나 적절한 prior를 두어 이 자유도를 처리한다. Stachniss의 *Graph-Based SLAM and Sparsity*는 그래프와 행렬의 관계를 설명하며, [추천 자료 안내](https://gisbi-kim.github.io/post/slam-textbooks/)에서 읽는 순서를 확인할 수 있다.

## 3. 틀린 loop도 최적화할 수 있다

1층 복도와 2층 복도가 비슷해서 같은 장소로 연결했다고 하자. Solver가 낮은 비용을 찾더라도 지도 두 층이 붙어 버릴 수 있다. 최적화는 입력 제약이 참인지 자동으로 보증하지 않는다.

학습할 때는 시스템을 다음 두 질문으로 나누자. “왜 이 제약을 만들었나?”는 front-end 질문이고, “주어진 제약들을 어떻게 조정했나?”는 back-end 질문이다.

## 확인 질문

정보 행렬을 크게 설정한 틀린 loop closure는 왜 특히 위험할까?

**확인:** 그 제약을 강하게 신뢰한다는 뜻이기 때문에 다른 타당한 제약을 희생하면서 맞출 수 있다. 기하 검증과 불확실성 설정을 함께 확인해야 한다.

## 원문 읽기

- KRoC Back-end: probability에서 least squares로 이어지는 부분. 로컬: `_resource/slam/kroc2026/04-backend-younggun-cho.pdf`.
- Stachniss (2016)는 추천 글의 설명과 링크를 참고한다. 기존 직접 PDF 주소는 현재 404를 반환하므로 로컬 자료에는 포함하지 않았다.
