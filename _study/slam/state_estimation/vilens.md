---
layout: study-chapter
title: "VILENS — 논문 리뷰"
description: "비전·IMU·LiDAR·다리 odometry를 그래프로 융합하고 다리 속도 bias를 추정하는 4족 상태 추정기다."
category: SLAM
series: state_estimation
importance: 6
permalink: /study/slam/state-estimation/vilens/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 비전·IMU·LiDAR·다리 odometry를 그래프로 융합하고 다리 속도 bias를 추정하는 4족 상태 추정기다.

| 항목        | 내용                                                                              |
| :---------- | :-------------------------------------------------------------------------------- |
| 논문        | VILENS: Visual, Inertial, Lidar, and Leg Odometry for All-Terrain Legged Robots   |
| 발표        | T-RO · 온라인 2022 / 권호 2023                                                    |
| 자료        | [논문·저자 자료](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                |
| 자료 확인일 | 2026-09-07                                                                        |

## 1. 해결하려는 문제

발의 미끄러짐, 지형 변형, 다리 유연성은 운동학 기반 속도에 오차를 만든다. 외부 센서도 어둠·먼지·특징 부족으로 퇴화할 수 있다.

## 2. 발표할 핵심 3개

1. **네 모달리티 강결합:** 영상·관성·LiDAR·다리 제약을 factor graph에서 결합한다.
2. **다리 속도 사전적분:** 다리 odometry의 속도를 시간 구간의 제약으로 구성한다.
3. **속도 bias 추정:** 선속도 bias를 상태에 추가하고 다른 모달리티와의 융합으로 추정한다.

기술 요약 근거: [논문·저자 설명](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
관절·다리 속도 → 속도 사전적분 factor + 비전·LiDAR·IMU factor → 상태·속도 bias 추정
```

bias는 다리 속도의 체계적인 오차를 설명하기 위한 상태다. 각 발의 미끄러짐을 완벽히 판별하는 접촉 분류기와 동일시하지 않는다. 이 논문의 중심은 다중 센서 odometry다.

## 4. 실험 결과와 해석

여러 ANYmal에서 총 2시간·1.8 km의 실험을 보고한다. 느슨한 돌, 경사, 진흙 및 어둡고 먼지 많은 환경이 포함된다. [출처](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf)

외부 센서와의 결합이 bias 관측에 중요하다. 모든 센서가 동시에 약해질 때의 거동과 다른 로봇으로 옮기는 비용은 별도로 평가해야 한다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. Vision60에서 timestamp가 있는 관절각·관절속도·접촉 정보를 받을 수 있는가?
2. 미끄러짐을 속도 bias로 설명할 수 있는 시간 규모는 어느 정도인가?
3. 센서 ablation에서 다리 factor와 bias 상태의 효과가 분리되어 있는가?

**제안 실험:** 미끄러지는 구간과 정상 접촉 구간을 표시한 로그로 다리 속도와 외부 기준 속도를 비교한다. bias 추가 전후의 속도·높이 오차를 구간별로 기록한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

VILENS가 다리 odometry에 단순한 고정 공분산만 주는 대신 선속도 bias를 상태로 추정하는 이유는 무엇인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

지형 변형, 다리 유연성, 미끄러짐이 지속적·방향성 있는 속도 오차를 만들 수 있기 때문이다. 고정 공분산은 신뢰도를 낮출 뿐 체계적 오차를 명시적으로 보정하지 못한다. 다른 센서 제약이 있어야 bias와 실제 운동을 구별할 관측 가능성이 생긴다.

</details>

### Q2. 수학·추론

측정 모델 \(z_v=v+b_v+n\)만 있고 \(v\)와 \(b_v\)를 모두 미지수로 두면 왜 둘을 분리하기 어려운가? 어떤 추가 정보가 필요한가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

한 측정은 \(v+b_v\)의 합만 제한하므로 \(v\leftarrow v+\Delta, b_v\leftarrow b_v-\Delta\)가 같은 측정을 만든다. 따라서 rank가 부족하다. IMU 동역학, LiDAR·visual pose 변화, bias random-walk prior 같은 시간·외부 제약이 필요하다. 단순히 센서 수가 많다는 답보다 관측 Jacobian의 null space를 설명해야 한다.

</details>

### Q3. 시스템·디버깅

접촉 중인 한 발이 미끄러질 때 모든 다리의 속도를 평균하면 어떤 문제가 생기는가? 강건화 방법을 제안하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

outlier인 발이 몸통 속도 추정을 오염시킨다. 발별 contact probability와 innovation을 유지하고 gating, robust loss, 발별 covariance 조절 또는 consistency 검사로 영향력을 낮출 수 있다. 실제 접촉 센서가 없으면 torque·kinematics 기반 접촉 추정의 지연과 오검출도 평가해야 한다.

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

**이어 읽기:** [LIJO 리뷰]({{ '/study/slam/state-estimation/lijo/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
