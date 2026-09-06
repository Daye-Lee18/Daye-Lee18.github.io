---
layout: study-chapter
title: "LOAM — 논문 리뷰"
description: "고주파 odometry와 저주파 정밀 mapping을 분리해 LiDAR 실시간 추정을 가능하게 한 계보의 출발점이다."
category: SLAM
series: state_estimation
importance: 9
permalink: /study/slam/state-estimation/loam/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 고주파 odometry와 저주파 정밀 mapping을 분리해 LiDAR 실시간 추정을 가능하게 한 계보의 출발점이다.

| 항목        | 내용                                                                                           |
| :---------- | :--------------------------------------------------------------------------------------------- |
| 논문        | LOAM: Lidar Odometry and Mapping in Real-time                                                  |
| 발표        | RSS 2014                                                                                       |
| 자료        | [논문·저자 자료](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                             |
| 자료 확인일 | 2026-09-07                                                                                     |

## 1. 해결하려는 문제

스캔 한 장을 취득하는 동안 센서가 움직이면 점마다 다른 pose에서 측정된다. 운동 추정과 정밀 매핑을 모두 빠르게 처리해야 한다.

## 2. 발표할 핵심 3개

1. **두 단계 처리:** 빠른 운동 추정과 더 정밀한 맵 정합을 서로 다른 주기로 수행한다.
2. **기하 특징 정합:** 모서리와 평면의 특징점을 활용해 움직임을 추정한다.
3. **계산 예산 분리:** odometry가 빠른 변화를 따라가는 동안 mapping이 더 정밀한 등록을 수행한다.

기술 요약 근거: [논문·저자 설명](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
LiDAR 스캔 → 특징 추출 → 빠른 odometry·운동 보정 → 정밀 mapping
```

각 점이 다른 시각에 수집된다는 문제가 왜 이후 LIO에서 IMU와 deskew를 중요하게 만드는지 연결해 읽는다. 원 논문은 IMU 보조 사용도 설명하므로 LiDAR-only가 유일한 구성이라고 단정하지 않는다.

## 4. 실험 결과와 해석

저자들은 여러 실험과 KITTI에서 평가하고 실시간성과 낮은 drift를 보고한다. 당시 비교 대상·센서와 현재 온보드 환경을 구분해 읽는다. [출처](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time)

LiDAR 중심의 기하 정합은 충분한 구조와 적절한 운동 추정에 의존한다. 이 논문의 두 단계 매핑을 전역 루프클로저와 동일시하지 않는다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. odometry와 mapping이 각각 추정하는 변환은 무엇인가?
2. 스캔 내 운동 보정에 필요한 속도는 어디에서 얻는가?
3. Vision60의 착지처럼 운동이 빠르게 변하면 어떤 가정이 약해지는가?

**제안 실험:** 먼저 스캔 내 점의 시간 분포와 보행 중 회전량을 시각화한다. 직접 구현하기 전에 FAST-LIO2가 같은 문제에 IMU를 어떻게 쓰는지 비교한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

LOAM에서 odometry와 mapping의 주기를 분리한 이유를 정확도와 계산량 관점에서 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

odometry는 적은 계산으로 빠르게 속도·운동을 따라가고 mapping은 더 많은 점과 반복으로 느리지만 정밀하게 등록한다. 두 결과가 서로 보완해 실시간성과 낮은 drift를 절충한다. 이 mapping을 전역 루프클로저와 동일시하면 안 된다.

</details>

### Q2. 수학·추론

point-to-plane 잔차 \(r=n^\top(Rp+t-q)\)와 point-to-line 거리의 차이를 설명하고, 각각 어떤 특징에 쓰이는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

평면 잔차는 변환된 점과 평면 기준점 \(q\)의 차이를 법선 \(n\)에 투영한 scalar다. 선 잔차는 점과 선 방향 \(d\) 사이의 수직 성분, 예를 들어 \(\|(I-dd^\top)(Rp+t-q)\|\)로 쓸 수 있다. planar feature에는 전자, edge feature에는 후자를 사용한다.

</details>

### Q3. 시스템·디버깅

보행 중 한 스캔 안에서 각속도가 일정하지 않다면 constant-velocity deskew가 만드는 지도 오류를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

실제 pose interpolation과 가정한 pose가 달라 점이 시간에 따라 체계적으로 잘못 변환된다. 평면이 휘거나 이중으로 보이고, 잘못된 대응점이 다음 pose에도 bias를 준다. 점별 timestamp와 고주파 IMU 기반 trajectory를 사용하고 착지 구간의 deskew 전후를 시각화해야 한다.

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

**이어 읽기:** [LIO-SAM 리뷰]({{ '/study/slam/state-estimation/lio-sam/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
