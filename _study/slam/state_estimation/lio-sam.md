---
layout: study-chapter
title: "LIO-SAM — 논문 리뷰"
description: "LiDAR·관성 추정을 factor graph로 구성해 GPS와 루프 제약을 함께 다루는 smoothing 기반 대조군이다."
category: SLAM
series: state_estimation
importance: 3
permalink: /study/slam/state-estimation/lio-sam/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** LiDAR·관성 추정을 factor graph로 구성해 GPS와 루프 제약을 함께 다루는 smoothing 기반 대조군이다.

| 항목        | 내용                                                                                                    |
| :---------- | :------------------------------------------------------------------------------------------------------ |
| 논문        | LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping                              |
| 발표        | IROS 2020                                                                                               |
| 자료        | [논문·저자 자료](https://arxiv.org/abs/2007.00258) · [공식 구현](https://github.com/TixiaoShan/LIO-SAM) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                                      |
| 자료 확인일 | 2026-09-07                                                                                              |

## 1. 해결하려는 문제

관성 예측과 LiDAR 정합을 연결하면서 과거 상태에 들어오는 상대·절대 제약도 수용해야 한다.

## 2. 발표할 핵심 3개

1. **IMU 사전적분:** 스캔 운동 보정과 LiDAR 정합 초기값에 관성 정보를 사용한다.
2. **로컬 키프레임 정합:** 새 스캔을 제한된 과거 키프레임 집합에 정합해 계산량을 관리한다.
3. **그래프 기반 융합:** LiDAR odometry, GPS, loop closure를 factor로 표현해 궤적을 보정한다.

기술 요약 근거: [논문·저자 설명](https://arxiv.org/abs/2007.00258).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
LiDAR + IMU → 사전적분·deskew → 특징 기반 로컬 정합 → 그래프 갱신 ← GPS·루프 제약
```

LiDAR 정합 결과가 IMU bias 추정에도 사용된다. 공식 구현은 IMU 추정 그래프와 매핑 그래프를 구분한다. 모든 원시 센서 잔차를 무한히 커지는 단일 그래프에 넣는 구조로 이해하지 않는다.

## 4. 실험 결과와 해석

논문은 세 플랫폼에서 여러 규모·환경의 데이터를 평가한다. 리뷰에서는 GPS·루프 사용 여부가 같은 실험끼리 비교해야 한다. 공식 저장소의 루프 구현은 proof of concept로 설명되어 있다. [출처](https://arxiv.org/abs/2007.00258) · [구현 설명](https://github.com/TixiaoShan/LIO-SAM)

루프 제약을 추가할 수 있다는 사실은 잘못된 장소 매칭까지 자동으로 해결한다는 뜻이 아니다. 센서 메시지의 점별 시간과 ring 정보 등 구현 요구조건도 확인해야 한다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. FAST-LIO2와 비교할 때 루프·GPS를 끈 로컬 오차와 켠 전역 오차를 나누었는가?
2. IMU 그래프의 초기화·재설정이 출력 pose의 연속성에 어떤 영향을 주는가?
3. Vision60 LiDAR 메시지를 공식 구현의 입력 형식으로 변환할 수 있는가?

**제안 실험:** 같은 재방문 로그에서 루프 비활성·활성 결과를 비교한다. 루프 전후 전역 오차와 pose jump를 함께 기록해 제어용 연속 상태와 지도용 보정 상태의 용도를 나눈다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

LIO-SAM이 factor graph를 쓴다는 사실만으로 루프클로저가 강건하다고 말할 수 없는 이유는 무엇인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

그래프는 들어온 제약을 최적화하는 표현이다. 장소인식의 false positive, 상대 pose 검증, robust loss, 잘못된 제약 제거는 별도의 문제다. 오검출 제약이 강하면 그래프는 오히려 전체 궤적을 잘못 변형할 수 있다. 후보 생성–기하 검증–그래프 최적화를 구분해야 한다.

</details>

### Q2. 수학·추론

IMU preintegration factor의 잔차가 대략 \(r=[r_R,r_v,r_p,r_b]\)이고 공분산이 \(\Sigma\)일 때 factor의 비용함수를 쓰고, bias를 다시 선형화해야 하는 이유를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

비용은 \(J\_{imu}=r^\top\Sigma^{-1}r\)이다. 사전적분된 회전·속도·위치는 적분 구간의 gyro/accel bias 가정에 의존한다. bias 추정치가 바뀌면 잔차와 Jacobian도 바뀌므로 1차 bias 보정 또는 재적분·재선형화가 필요하다. Mahalanobis 가중치가 센서 불확실성을 반영한다는 설명까지 포함하면 좋다.

</details>

### Q3. 시스템·디버깅

루프클로저 후 지도는 맞지만 제어기가 순간적으로 불안정해졌다. 상태 인터페이스를 어떻게 설계하겠는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

연속적인 local odometry frame과 전역 보정 map frame을 분리한다. 제어기는 짧은 시간에 연속인 odom→base 상태를 쓰고, SLAM은 map→odom 변환을 갱신한다. 전역 보정량을 즉시 적용할지 완만히 반영할지도 소비자별로 정한다. timestamp와 transform tree의 단일 소유자를 명확히 해야 한다.

</details>

### Q4. 코드 리뷰·구현

**GitHub:** [공식 또는 저자 연결 저장소](https://github.com/TixiaoShan/LIO-SAM)

공식 구현의 `imageProjection.cpp`, `featureExtraction.cpp`, `mapOptimization.cpp`, `imuPreintegration.cpp` 사이에서 한 LiDAR 스캔이 최종 pose가 되기까지 데이터 흐름을 설명하고, 잘못된 `time` 필드를 가장 먼저 검출할 위치를 정하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

`imageProjection.cpp`가 cloud deskew와 range image 투영을 수행하고, `featureExtraction.cpp`가 edge·surface 특징을 만든다. `mapOptimization.cpp`는 키프레임·factor graph·map 정합을 관리하고, `imuPreintegration.cpp`는 IMU 주파수 상태와 bias를 갱신한다. 점별 `time`은 deskew 전에 유효 범위와 단조성을 검사해야 한다. 10 Hz LiDAR라면 상대 시간이 대략 한 scan period 안에 있는지 확인하고 위반 시 명확히 실패시켜야 이후 모듈의 이상 pose로 숨지 않는다.

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
