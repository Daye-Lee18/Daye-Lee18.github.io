---
layout: study-chapter
title: "Legolas — 논문 리뷰"
description: "다리·관성 센서만으로 odometry를 학습해 외부 센서 추적이 어려운 상황을 다루는 연구다."
category: SLAM
series: state_estimation
importance: 13
permalink: /study/slam/state-estimation/legolas/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 다리·관성 센서만으로 odometry를 학습해 외부 센서 추적이 어려운 상황을 다루는 연구다.

| 항목        | 내용                                                                   |
| :---------- | :--------------------------------------------------------------------- |
| 논문        | Legolas: Deep Leg-Inertial Odometry                                    |
| 발표        | 제8회 CoRL · PMLR 270, 2025                                            |
| 자료        | [논문·저자 자료](https://proceedings.mlr.press/v270/wasserman25a.html) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                     |
| 자료 확인일 | 2026-09-07                                                             |

## 1. 해결하려는 문제

분석적 다리 odometry는 로봇별 모델과 튜닝이 필요하고, 기존 학습 방식은 고품질 실세계 궤적 수집과 분포 변화에 영향을 받는다.

## 2. 발표할 핵심 3개

1. **학습 기반 leg-inertial odometry:** 다리와 IMU 신호에서 이동을 추정한다.
2. **실세계 학습 궤적 수집 없이 학습:** 기존의 실측 궤적 데이터 의존성을 줄이는 접근이다.
3. **실제 로봇 평가:** 두 4족 플랫폼의 실내·실외 환경에서 배포 결과를 제시한다.

기술 요약 근거: [논문·저자 설명](https://proceedings.mlr.press/v270/wasserman25a.html).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
다리 센서 + IMU → 학습된 odometry 모델 → 상대 운동·궤적 추정
```

외부 환경 정합 없이 proprioception으로 이동을 추정한다는 점이 LIO·LIVO와 다르다. 네트워크 구조, 입력 시간 창, 출력 표현과 학습 손실은 원문 정독 시 별도로 도식화한다.

## 4. 실험 결과와 해석

공식 초록은 실내 장면에서 상대 pose 오차가 분석적 필터 기준보다 73%, 실세계 behavioral cloning 기준보다 87.5% 낮았다고 보고한다. [출처](https://proceedings.mlr.press/v270/wasserman25a.html)

해당 상대 개선률을 FAST-LIO2와의 비교로 바꾸어 읽으면 안 된다. 새로운 로봇·마찰·보행 분포에서의 일반화는 별도 문제다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. 학습 입력의 관절 순서·단위·주기와 Vision60 센서가 일치하는가?
2. 학습 환경이 실제 슬립·충격·센서 지연을 어느 범위까지 포함하는가?
3. 추정 불확실성 없이 LIO에 결합한다면 오류를 과신할 가능성은 없는가?

**제안 실험:** 학습 모델 단독 odometry의 정상 보행·슬립 구간 상대 오차를 먼저 측정한다. LIO 융합은 그 후 단계로 두고 새로운 factor의 공분산을 어떻게 정할지 검토한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

학습 기반 leg-inertial odometry가 새로운 로봇에 바로 일반화하기 어려운 이유를 입력 표현과 동역학 관점에서 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

관절 수·순서·축, 링크 길이, actuator dynamics, gait, 마찰, 센서 bias·주기가 달라 같은 입력 패턴의 물리적 의미가 바뀐다. normalization과 augmentation만으로 invariance가 보장되지 않는다. morphology-aware 표현, adaptation과 불확실성 평가가 필요하다.

</details>

### Q2. 수학·추론

모델이 시간창 \(x*{t-k:t}\)에서 상대 변환 \(\hat T*{t,t+1}\)을 예측할 때 SE(3) loss를 하나 쓰고, translation·rotation scale 문제를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

예를 들어 \(\xi=\mathrm{Log}(T\_{gt}^{-1}\hat T)\), \(L=\xi^\top W\xi\)를 쓸 수 있다. \(\xi=[\rho,\phi]\)에서 m와 rad의 scale이 달라 \(W\) 또는 learned uncertainty가 필요하다. 회전 표현의 discontinuity와 긴 rollout에서 누적되는 상관 오차도 별도로 평가해야 한다.

</details>

### Q3. 시스템·디버깅

학습 odometry를 LIO factor로 추가했더니 전체 정확도가 나빠졌다. 가능한 원인과 안전한 결합법은 무엇인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

분포 밖 상황에서 bias가 커졌는데 covariance를 작게 주어 과신했을 수 있다. 시간 상관된 출력을 독립 factor처럼 반복 사용했을 수도 있다. innovation gating, OOD·uncertainty 추정, covariance calibration, 낮은 초기 가중치와 failure fallback을 사용하고 sensor ablation으로 원인을 분리한다.

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

**이어 읽기:** [VILENS 리뷰]({{ '/study/slam/state-estimation/vilens/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
