---
layout: study-chapter
title: "GLIM — 논문 리뷰"
description: "시간 창 안의 상태와 서브맵 간 정합을 최적화하고 GPU로 계산량을 처리하는 매핑 프레임워크다."
category: SLAM
series: state_estimation
importance: 8
permalink: /study/slam/state-estimation/glim/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 시간 창 안의 상태와 서브맵 간 정합을 최적화하고 GPU로 계산량을 처리하는 매핑 프레임워크다.

| 항목        | 내용                                                                                             |
| :---------- | :----------------------------------------------------------------------------------------------- |
| 논문        | GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors      |
| 발표        | Robotics and Autonomous Systems 2024                                                             |
| 자료        | [논문·저자 자료](https://arxiv.org/abs/2407.10344) · [공식 구현](https://github.com/koide3/glim) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                               |
| 자료 확인일 | 2026-09-07                                                                                       |

## 1. 해결하려는 문제

현재 스캔의 기하 제약이 일시적으로 약할 때 과거 상태와 정합 관계를 활용해야 한다. 풍부한 제약을 최적화하려면 계산 비용도 관리해야 한다.

## 2. 발표할 핵심 3개

1. **Fixed-lag smoothing:** 제한된 시간 창의 상태를 함께 추정한다.
2. **GPU scan matching factor:** 점군 정합 오차 계산의 병렬성을 활용한다.
3. **전역 등록 오차 최소화:** 전체 맵의 서브맵 간 등록 오차를 직접 최적화한다.

기술 요약 근거: [논문·저자 설명](https://arxiv.org/abs/2407.10344).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
Range + IMU (+ 카메라 제약) → 로컬 fixed-lag smoothing → 서브맵 → 전역 정합 최적화
```

논문은 다중 카메라 특징 제약의 강결합도 설명한다. 로컬 시간 창과 전역 서브맵 최적화가 각각 어떤 상태를 보유하는지 구분해 읽는다.

## 4. 실험 결과와 해석

저자들은 몇 초 동안 range data가 완전히 퇴화하는 상황을 다루는 능력과 GPU를 이용한 실시간 처리를 보고한다. 이런 결과가 무제한 퇴화나 모든 GPU에서의 실시간성을 보장하지는 않는다. [출처](https://arxiv.org/abs/2407.10344) · [구현 설명](https://github.com/koide3/glim)

필터 방식보다 많은 계산을 사용하는 설계이므로 온보드 GPU의 다른 작업과 경쟁할 수 있다. 논문 GLIM은 2024년이며 기반이 된 2022년 연구와 구분한다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. 시간 창 길이를 늘릴 때 정확도·메모리·지연이 어떻게 변하는가?
2. Vision60의 GPU가 인지·제어 작업을 함께 수행하는가?
3. 기하 퇴화 회복에 카메라와 시간 창이 각각 얼마나 기여하는가?

**제안 실험:** 동일 퇴화 로그에서 FAST-LIO2와 비교하고 구간별 오차, 지연 상위 백분위수, GPU 메모리를 기록한다. 카메라 사용 여부를 맞춘 뒤 비교한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

fixed-lag smoothing이 EKF보다 일시적인 LiDAR 퇴화에 유리할 수 있는 이유와 비용을 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

시간 창의 여러 상태와 관측을 함께 다시 선형화하므로 퇴화 구간 전후의 정보가 내부 상태를 연결할 수 있다. 반면 상태·factor 수, 메모리, 선형계 풀이 비용과 지연이 증가한다. 창 밖 상태의 marginalization이 만든 prior의 일관성도 고려해야 한다.

</details>

### Q2. 수학·추론

비선형 least squares \(\min*\delta\sum_i\|r_i(x\boxplus\delta)\|*{\Sigma_i^{-1}}^2\)를 1차 선형화한 normal equation을 쓰고, 퇴화와의 관계를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

\(H\delta=-g\), \(H=\sum_iJ_i^\top\Sigma_i^{-1}J_i\), \(g=\sum_iJ_i^\top\Sigma_i^{-1}r_i\)이다. \(H\)의 작은 고유값은 해당 상태 방향을 제약하는 정보가 약함을 뜻한다. damping은 수치 안정성을 줄 수 있지만 관측 정보를 새로 만들지는 않는다.

</details>

### Q3. 시스템·디버깅

평균 실행 시간은 실시간인데 Vision60 제어가 간헐적으로 deadline을 놓친다. GPU 기반 SLAM 관점에서 무엇을 측정할 것인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

kernel·memory transfer·synchronization별 시간, p95/p99와 최악 지연, GPU memory peak, queue depth, 인지·학습 작업과의 contention, thermal throttling을 측정한다. 평균 FPS만 보면 burst와 blocking을 놓친다. 제어와 SLAM의 stream·priority·resource partition도 확인한다.

</details>

### Q4. 코드 리뷰·구현

**GitHub:** [공식 또는 저자 연결 저장소](https://github.com/koide3/glim)

GLIM에 새로운 속도 factor를 추가한다고 하자. factor 구현 외에 registration, config, state timestamp, Jacobian 검증과 실시간 성능을 어떻게 확인하겠는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

기존 factor가 생성·등록되는 경로와 optimizer가 보유하는 state key·timestamp convention을 먼저 따른다. noise model과 robust kernel을 config로 노출하고 센서 측정 시각을 어떤 두 상태에 연결할지 명시한다. analytic Jacobian은 manifold perturbation convention을 맞춰 finite difference와 비교한다. factor on/off의 동일 bag A/B 테스트에서 trajectory error, innovation, graph size, solve-time p95/p99를 측정하고 marginalization 뒤에도 dangling key가 없는지 검사한다.

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
