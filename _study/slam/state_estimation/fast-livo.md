---
layout: study-chapter
title: "FAST-LIVO — 논문 리뷰"
description: "LiDAR·관성·영상을 sparse-direct 방식으로 결합하는 FAST-LIVO2의 선행 연구다."
category: SLAM
series: state_estimation
importance: 11
permalink: /study/slam/state-estimation/fast-livo/
---

[← 상태 추정 논문 비교]({{ '/study/slam/lio/' | relative_url }})

> **한 문장 요약:** LiDAR·관성·영상을 sparse-direct 방식으로 결합하는 FAST-LIVO2의 선행 연구다.

| 항목 | 내용 |
|:---|:---|
| 논문 | FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry |
| 발표 | IROS 2022 |
| 자료 | [논문·저자 자료](https://github.com/hku-mars/FAST-LIVO) |
| 정리 상태 | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가 |
| 자료 확인일 | 2026-09-07 |

## 1. 해결하려는 문제

LiDAR 기하와 이미지 정보가 서로 다른 장점을 가지므로 이를 같은 odometry 과정에서 활용하고자 한다.

## 2. 발표할 핵심 3개

1. **LIO와 VIO 결합:** LiDAR-inertial·visual-inertial 두 하위 시스템을 강하게 연결한다.
2. **Sparse-direct 비전:** 영상의 sparse-direct 접근을 사용한다.
3. **다중 센서 odometry:** 영상과 LiDAR가 제공하는 상보적인 정보를 함께 활용한다.

기술 요약 근거: [논문·저자 설명](https://github.com/hku-mars/FAST-LIVO).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
LiDAR + IMU → LIO 하위 시스템 ↔ VIO 하위 시스템 ← 이미지
```

공식 저장소는 두 개의 tightly coupled direct odometry 하위 시스템으로 구조를 설명한다. FAST-LIVO2의 통합 voxel map·순차 갱신 세부사항을 이 논문에 그대로 소급하지 않는다.

## 4. 실험 결과와 해석

공식 저장소에서 논문·코드·데이터 접근 경로를 제공한다. 이 페이지는 저자 설명 기반의 입문 리뷰이며 데이터셋별 성능 수치는 정독 후 기록한다. [출처](https://github.com/hku-mars/FAST-LIVO)

카메라 동기화와 외부 파라미터를 맞추는 과정도 재현의 일부다. 후속 버전의 개선 주장을 선행 논문의 실험 결과로 기록하지 않는다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. LIO와 VIO가 공유하는 상태·맵 정보는 정확히 무엇인가?
2. 영상 잔차를 계산할 점 또는 패치는 어떻게 고르는가?
3. FAST-LIVO2의 개선 중 맵 표현 변화와 갱신 방식 변화는 각각 무엇인가?

**제안 실험:** 동일한 센서 로그를 재현할 수 있는지 먼저 확인한다. 영상 노출·블러 상태를 표시해 영상 정보가 유효한 구간과 그렇지 않은 구간을 구분한다.

## 6. 정독·발표 기록

위 요약을 출발점으로 원문의 수식·그림·실험 표를 확인한 뒤 직접 채우는 공간이다. 아직 수행하지 않은 재현 결과는 논문 결과와 구분해 남긴다.

| 기록할 항목 | 개인 리뷰 메모 |
|:---|:---|
| 상태·입력·출력 | 미작성 — 좌표계, 단위, 센서 주기까지 기록 |
| 핵심 수식 | 미작성 — 식 번호, 변수 의미, 가정과 잔차를 설명 |
| 대표 그림 | 미작성 — 그림 번호와 데이터 흐름을 본인의 말로 설명 |
| 실험 근거 | 미작성 — 표·그림 번호, 데이터셋, baseline, 지표와 조건 |
| Ablation | 미작성 — 어떤 요소를 제거했고 무엇이 바뀌었는지 기록 |
| 실패 사례·한계 | 미작성 — 저자 보고와 자신의 추론을 구분 |
| 코드·재현 | 미작성 — 버전, 설정, 로그, 장치, 측정 결과 |
| 최종 판단 | 미작성 — Vision60에서 채택·보류할 이유 |

- [ ] 핵심 기여 3개를 원문 근거와 함께 설명할 수 있다.
- [ ] 상태와 관측이 어떻게 연결되는지 설명할 수 있다.
- [ ] 실험 결과와 Vision60 적용 가설을 구분했다.

**이어 읽기:** [FAST-LIVO2 리뷰]({{ '/study/slam/state-estimation/fast-livo2/' | relative_url }}) · [전체 비교표]({{ '/study/slam/lio/' | relative_url }})

