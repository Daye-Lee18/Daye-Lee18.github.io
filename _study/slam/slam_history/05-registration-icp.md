---
layout: study-chapter
title: "Chapter 5. ICP와 점군 정합"
description: "두 관측을 맞추는 문제에서 반복 최적화를 이해하기."
importance: 5
category: SLAM
series: slam_history
permalink: /study/slam/history/05-registration-icp/
---
> **목표:** correspondence와 pose update를 구분한다.  
> **학습량:** 15분. Chapter 4의 좌표 변환을 사용한다.

## 1. 두 점군이 서로 어긋나 있다

다른 위치에서 얻은 점군을 같은 좌표계로 맞추려면 회전과 이동이 필요하다. Point-to-point ICP의 대표적인 목적함수는 다음과 같다.

$$
\min_{R,t}\sum_i\|Rp_i+t-q_{c(i)}\|^2
$$

$c(i)$는 source 점 $p_i$에 대응시킨 target 점의 번호다. 기본 흐름은 **현재 변환으로 대응점을 찾고 → 대응을 고정해 변환을 개선하고 → 다시 대응을 찾는 것**이다. [Giseop Kim의 SLAM Back-end 자료 안내](https://gisbi-kim.github.io/post/slam-textbooks/)가 추천하는 Grisetti의 *From Least-Squares to ICP*는 이 작은 문제에서 residual과 Jacobian을 연결한다.

## 2. 오차를 줄이는 것만으로 충분할까?

반복해서 같은 모양이 나오는 복도에서 옆 기둥을 대응점으로 잡아도 작은 오차를 만들 수 있다. 초기값, 겹치는 영역, 잘못된 점 제거가 중요한 이유다.

Point-to-plane은 점과 대응 평면 사이의 법선 방향 거리를 사용한다. 벽 하나만 보이면 벽을 따라 미끄러지는 움직임은 이 거리로 잘 구분하지 못한다. 측정이 어떤 방향을 제약하는지 확인해야 한다. 점군 정합의 발전은 [KRoC 정합 강연](https://drive.google.com/file/d/1fwaHF77iwTxcDVl1s1h8BcCZE0EvPmUC/view)으로 확장해서 읽는다.

## 3. 1차원으로 한 번 계산하기

대응이 이미 알려져 있고 회전이 없다고 하자. Source가 `[0, 1, 2]`, target이 `[2, 3, 4]`라면 residual은 $r_i=p_i+t-q_i$다. 제곱합을 최소화하는 이동은 $t=2$다.

그런데 마지막 target을 40으로 잘못 매칭하면 최적 이동은 $(2+2+38)/3=14$가 된다. 계산은 정확해도 대응이 틀리면 추정은 틀린다. Robust loss와 correspondence 검증은 서로 보완한다.

## 확인 질문

ICP가 수렴했다는 로그가 “실제 위치를 찾았다”는 뜻일까?

**확인:** 아니다. 선택한 목적함수와 초기값 아래에서 변화가 작아졌다는 의미일 수 있다. 다른 정합 해, 잘못된 대응, 관측 불가능한 방향이 남아 있을 수 있다.

## 원문 읽기

- KRoC 정합 강연: ICP에서 robust/global/learning 접근으로 연결되는 부분. 로컬: `_resource/slam/kroc2026/06-registration-hyungtae-lim.pdf`.
- Grisetti (2016)는 추천 글의 설명과 링크를 참고한다. 기존 직접 PDF 주소는 현재 404를 반환하므로 로컬 자료에는 포함하지 않았다.
