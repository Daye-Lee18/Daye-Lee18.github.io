---
layout: study-chapter
title: "FAST-LIO — 논문 리뷰"
description: "LiDAR 특징점과 IMU를 반복 EKF로 결합하며, 많은 관측을 효율적으로 처리하는 FAST-LIO2의 기반이다."
category: SLAM
series: state_estimation
importance: 10
permalink: /study/slam/state-estimation/fast-lio/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** LiDAR 특징점과 IMU를 반복 EKF로 결합하며, 많은 관측을 효율적으로 처리하는 FAST-LIO2의 기반이다.

| 항목        | 내용                                                                                               |
| :---------- | :------------------------------------------------------------------------------------------------- |
| 논문        | FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter |
| 발표        | arXiv 2020                                                                                         |
| 자료        | [논문·저자 자료](https://arxiv.org/abs/2010.08196)                                                 |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                                 |
| 자료 확인일 | 2026-09-07                                                                                         |

## 1. 해결하려는 문제

많은 LiDAR 관측을 강결합 필터에서 처리할 때 Kalman gain 계산 비용이 커질 수 있다.

## 2. 발표할 핵심 3개

1. **Tightly coupled 융합:** LiDAR 특징점과 관성 정보를 같은 추정 과정에서 사용한다.
2. **Iterated EKF:** 반복 갱신으로 비선형 관측을 처리한다.
3. **효율적인 gain 계산:** 관측 차원보다 상태 차원에 기반하는 계산식을 사용해 비용을 줄인다.

기술 요약 근거: [논문·저자 설명](https://arxiv.org/abs/2010.08196).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
LiDAR 특징점 + IMU → 운동 예측 → 반복 필터 갱신 → 상태 추정
```

이 논문에서 먼저 필터의 상태·관측·갱신을 이해한 뒤 FAST-LIO2가 특징 추출과 맵 관리에서 무엇을 바꾸는지 읽는다. 빠른 필터와 좋은 맵 자료구조는 다른 설계 요소다.

## 4. 실험 결과와 해석

저자들은 UAV 온보드 환경에서 한 스캔의 유효 특징점 1,200개 이상을 융합하고 반복 갱신 전체를 25 ms 이내에 처리한 사례를 보고한다. [출처](https://arxiv.org/abs/2010.08196)

이 수치는 해당 구현·장치의 결과다. 관측이 늘어도 전체 파이프라인 비용이 항상 일정하다는 의미는 아니다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. 상태 차원과 관측 차원이 각각 무엇을 세는가?
2. 반복 중 재선형화하는 항과 고정하는 항은 무엇인가?
3. FAST-LIO2와 비교할 때 필터 개선과 맵 관리 개선을 분리할 수 있는가?

**제안 실험:** 현재 코드의 실행 시간을 IMU 처리, 대응점 탐색, 필터 갱신, 맵 갱신으로 나눠 기록한다. 병목이 실제로 gain 계산인지 확인한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

FAST-LIO와 FAST-LIO2의 기여를 필터, 관측 선택, 맵 자료구조 세 축으로 비교하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

FAST-LIO는 특징점과 IMU를 효율적인 iterated Kalman filter로 강결합하는 기반을 제시한다. FAST-LIO2는 이 필터 계보 위에서 raw-point direct 등록과 ikd-Tree 맵 관리를 핵심 기여로 추가한다. FAST-LIO2의 모든 개선을 필터 수식 변화로 설명하면 부정확하다.

</details>

### Q2. 수학·추론

관측 수 \(m\)이 상태 차원 \(n\)보다 매우 클 때 \(m\times m\) 행렬 역산 대신 information form을 쓰는 계산상 이유를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

표준 gain 식은 관측 공간의 \(HPH^\top+R\) 역산을 포함해 큰 \(m\)에서 부담이 크다. Woodbury identity 또는 information form을 이용하면 \(n\times n\) 상태 공간의 \(P^{-1}+H^\top R^{-1}H\)를 풀 수 있다. 복잡도 이점은 구조·희소성과 \(R\) 처리 방식에도 의존한다.

</details>

### Q3. 시스템·디버깅

필터 업데이트가 25 ms인데 전체 노드는 50 ms가 걸린다. 어디를 프로파일링해야 하는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

전처리·특징 추출, timestamp 정렬, deskew, nearest-neighbor 검색, 잔차 구성, 필터 반복, 맵 삽입·삭제, publish/copy를 분리한다. wall time과 CPU time, allocation, queue 대기를 함께 본다. 논문의 filter timing을 전체 pipeline latency와 비교하면 안 된다.

</details>

### Q4. 코드 리뷰·구현

**GitHub:** [공식 또는 저자 연결 저장소](https://github.com/hku-mars/FAST_LIO)

`src/laserMapping.cpp`의 한 주기 실행 시간이 길다. 코드를 어떤 단계로 계측하고, 최적화가 추정 결과를 바꾸지 않았음을 어떻게 확인하겠는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

동기화, IMU propagation·undistortion, downsampling, nearest-neighbor search, 잔차/Jacobian 구성, iterated update, ikd-Tree 삽입·삭제, publish를 각각 계측한다. 평균뿐 아니라 p95/p99와 처리 점 수를 함께 남긴다. 변경 전후 동일 bag에서 timestamp별 pose, 유효 잔차 수, map point 수를 비교하고 허용 오차를 명시한다. 병렬화한다면 공유 map과 correspondence buffer의 race도 thread sanitizer 또는 결정성 반복 실행으로 확인한다.

</details>

## 7. 정독·발표 기록

위 요약을 출발점으로 원문의 수식·그림·실험 표를 확인한 뒤 직접 채우는 공간이다. 아직 수행하지 않은 재현 결과는 논문 결과와 구분해 남긴다.

| 기록할 항목    | 개인 리뷰 메모                                         |
| :------------- | :----------------------------------------------------- |
| 상태·입력·출력 | 미작성 — 좌표계, 단위, 센서 주기까지 기록              |
| 핵심 수식      | 미작성 — 식 번호, 변수 의미, 가정과 잔차를 설명        |
| 대표 그림      | 미작성 — 그림 번호와 데이터 흐름을 본인의 말로 설명    |
| 실험 근거      | 미작성 — 표·그림 번호, 데이터셋, baseline, 지표와 조건 |
| Ablation       | 미작성 — 어떤 요소를 제거했고 무엇이 바뀌었는지 기록   |
| 실패 사례·한계 | 미작성 — 저자 보고와 자신의 추론을 구분                |
| 코드·재현      | 미작성 — 버전, 설정, 로그, 장치, 측정 결과             |
| 최종 판단      | 미작성 — Vision60에서 채택·보류할 이유                 |

- [ ] 핵심 기여 3개를 원문 근거와 함께 설명할 수 있다.
- [ ] 상태와 관측이 어떻게 연결되는지 설명할 수 있다.
- [ ] 실험 결과와 Vision60 적용 가설을 구분했다.

**이어 읽기:** [FAST-LIO2 리뷰]({{ '/study/slam/state-estimation/fast-lio2/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
