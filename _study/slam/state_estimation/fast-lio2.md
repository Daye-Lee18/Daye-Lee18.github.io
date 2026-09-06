---
layout: study-chapter
title: "FAST-LIO2 — 논문 리뷰"
description: "원시 LiDAR 점을 로컬 맵에 직접 정합하고 IMU와 반복 필터로 융합하는, 현재 Vision60 시스템의 기준선이다."
category: SLAM
series: state_estimation
importance: 2
permalink: /study/slam/state-estimation/fast-lio2/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 원시 LiDAR 점을 로컬 맵에 직접 정합하고 IMU와 반복 필터로 융합하는, 현재 Vision60 시스템의 기준선이다.

| 항목        | 내용                                               |
| :---------- | :------------------------------------------------- |
| 논문        | FAST-LIO2: Fast Direct LiDAR-inertial Odometry     |
| 발표        | T-RO 2022 · arXiv 2021                             |
| 자료        | [논문·저자 자료](https://arxiv.org/abs/2107.06829) |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가 |
| 자료 확인일 | 2026-09-07                                         |

## 1. 해결하려는 문제

스캔 패턴마다 특징 추출을 설계하고 커지는 점군 맵을 관리하면 계산량이 증가한다. FAST-LIO2는 정합 방식과 맵 자료구조를 함께 바꾼다.

## 2. 발표할 핵심 3개

1. **Direct 등록:** 미리 모서리·평면 특징점을 선별하지 않고 원시 점을 맵에 등록한다. 대응점과 기하 제약은 여전히 필요하다.
2. **반복 필터 융합:** FAST-LIO 계열의 효율적인 tightly coupled iterated Kalman filter를 기반으로 관성·LiDAR 정보를 결합한다.
3. **ikd-Tree:** 점 삽입·삭제·재균형과 downsampling을 지원해 이동 중 로컬 맵을 갱신한다. 논문이 강조하는 새 기여는 direct 등록과 이 자료구조다.

기술 요약 근거: [논문·저자 설명](https://arxiv.org/abs/2107.06829).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
LiDAR + IMU → 관성 전파·스캔 운동 보정 → 로컬 맵 대응점 → 반복 상태 갱신 → pose·맵 갱신
```

리뷰에서는 IMU가 만든 운동 예측과 LiDAR 정합이 그 예측을 수정하는 경로를 분리해 본다. 로컬 맵을 생성하지만 원 논문의 범위에 전역 루프클로저 백엔드는 포함되지 않는다.

## 4. 실험 결과와 해석

저자들은 공개 데이터 19개 시퀀스와 여러 LiDAR·플랫폼에서 평가했다. 초록은 특정 실험에서 최대 100 Hz odometry·mapping 및 1000 deg/s 회전 추정을 보고한다. 모든 센서와 장치에서 보장되는 수치로 해석하면 안 된다. [출처](https://arxiv.org/abs/2107.06829)

정확도와 속도는 환경의 기하 구조와 연산 조건에 의존한다. 루프 없는 장거리 누적 오차와 순간적인 추정 실패를 구분해야 한다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. Vision60 로그에서 점별 시각·IMU 시각·외부 파라미터는 어떻게 설정되어 있는가?
2. 단차 구간의 Z drift가 점군 왜곡, IMU 이상, 기하 제약 부족 중 무엇과 함께 나타나는가?
3. 후속 백엔드에 넘길 pose·점군의 좌표계와 시각은 무엇인가?

**제안 실험:** 같은 로그의 평지·회전·착지 구간을 분리해 상대 pose 오차, 높이 오차, 처리 지연을 기록한다. ground truth가 없으면 drift 수치를 정확도로 단정하지 말고 맵 중첩·반복 주행 차이를 보조 지표로 남긴다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

FAST-LIO2의 `direct`가 “ICP나 대응점 탐색을 하지 않는다”는 뜻인지 설명하고, 특징 기반 LIO와 비교하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

그 뜻이 아니다. FAST-LIO2는 hand-crafted edge/plane 특징 추출을 생략하고 원시 점을 map에 직접 등록한다. 각 점 주변의 map 이웃을 찾고 국소 평면을 구성해 point-to-plane 잔차를 만든다. 차이는 **관측을 만들 점을 미리 특징으로 선별하는가**에 가깝다. 좋은 답은 direct 등록, 대응점 탐색, 잔차 계산을 서로 구분한다.

</details>

### Q2. 수학·추론

점 \(p*i^L\)를 world frame으로 변환한 \(p_i^W=R*{WI}(R*{IL}p_i^L+t*{IL})+t\_{WI}\)와 평면 \(n_i^\top x+d_i=0\)가 있다. LiDAR 잔차를 쓰고, 어떤 상태에 대한 Jacobian이 퇴화할 수 있는지 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

잔차는 \(r*i=n_i^\top p_i^W+d_i\)이다. 작은 회전 섭동 \(\delta\theta\)에 대해 회전 Jacobian은 부호 convention에 따라 \(-n_i^\top R*{WI}[R_{IL}p_i^L+t_{IL}]\_\times\), 위치 Jacobian은 \(n_i^\top\) 꼴이다. 모든 법선이 비슷하면 법선에 수직인 평행이동이 약하게 관측되고, 회전도 점 분포와 법선이 충분히 다양하지 않으면 약해진다. 핵심은 Jacobian 또는 information matrix의 작은 고유값과 기하 퇴화를 연결하는 것이다.

</details>

### Q3. 시스템·디버깅

계단 착지 직후 Z가 튄다. 필터 튜닝 전에 어떤 로그를 어떤 순서로 확인할 것인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

점별 timestamp와 LiDAR–IMU 시간 기준, IMU clipping·drop, extrinsic, deskew 전후 점군, 잔차와 선택된 평면 법선, covariance 순서로 본다. 같은 raw log를 재생해 재현성을 확보하고 착지 전후를 분리한다. 단순히 process noise를 줄이면 출력은 매끄러워질 수 있지만 bias나 시간 오차를 숨기고 지연을 키울 수 있다.

</details>

### Q4. 코드 리뷰·구현

**GitHub:** [공식 또는 저자 연결 저장소](https://github.com/hku-mars/FAST_LIO)

저장소에서 새 LiDAR의 `PointCloud2`를 지원해야 한다. 점별 시간 필드의 단위가 기존 센서와 다를 때, 어느 처리 경로와 설정을 추적하고 어떤 회귀 테스트를 작성하겠는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

`config`의 `lidar_type`, `timestamp_unit`, topic 설정에서 시작해 `src/preprocess.cpp`의 센서별 callback·점 시간 변환, `src/laserMapping.cpp`의 measurement synchronization, `src/IMU_Processing.hpp`의 undistortion까지 추적한다. 내부 시간 단위를 하나로 정규화하고 scan 시작·끝 시각 및 점별 상대 시간이 단조인지 검사한다. 정지 bag, 일정 각속도 bag, 알려진 timestamp offset을 넣은 bag을 재생해 deskew 후 평면 두께와 pose를 비교한다. 단순히 컴파일되는지만 보는 테스트는 시간 단위 오류를 잡지 못한다.

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

**이어 읽기:** [FAST-LIO 리뷰]({{ '/study/slam/state-estimation/fast-lio/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
