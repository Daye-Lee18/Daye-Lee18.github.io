---
layout: study-chapter
title: "FAST-LIVO2 경량화 — 논문 리뷰"
description: "관측의 유용성과 맵 보존 범위를 조절해 FAST-LIVO2의 메모리·계산 비용을 줄이는 연구다."
category: SLAM
series: state_estimation
importance: 12
permalink: /study/slam/state-estimation/fast-livo2-resource-constrained/
---

[← 상태 추정 논문 비교]({{ '/study/slam/state-estimation/' | relative_url }})

> **한 문장 요약:** 관측의 유용성과 맵 보존 범위를 조절해 FAST-LIVO2의 메모리·계산 비용을 줄이는 연구다.

| 항목        | 내용                                                                                                               |
| :---------- | :----------------------------------------------------------------------------------------------------------------- |
| 논문        | FAST-LIVO2 on Resource-Constrained Platforms: LiDAR-Inertial-Visual Odometry with Efficient Memory and Computation |
| 발표        | 2025 · arXiv:2501.13876                                                                                            |
| 자료        | [논문·저자 자료](https://arxiv.org/abs/2501.13876)                                                                 |
| 정리 상태   | 입문 리뷰 초안 · 개인 정독·재현 기록은 아래에 추가                                                                 |
| 자료 확인일 | 2026-09-07                                                                                                         |

## 1. 해결하려는 문제

LIVO가 정확해도 온보드 장치의 계산·메모리 예산을 넘으면 지속적으로 운용하기 어렵다.

## 2. 발표할 핵심 3개

1. **적응적 영상 선택:** 퇴화를 고려해 사용할 영상 프레임을 선택한다.
2. **맵 메모리 관리:** 로컬 통합 LiDAR-visual map과 장기 visual map을 함께 사용한다.
3. **효율·정확도 절충:** ESIKF 순차 갱신 구조에서 자원 비용을 줄인다.

기술 요약 근거: [논문·저자 설명](https://arxiv.org/abs/2501.13876).

## 3. 동작 구조

아래는 이해를 위한 개념 흐름이며 구현의 모든 스레드·갱신 주기를 나타내지는 않는다.

```text
센서 입력 → 퇴화 기반 영상 선택 → 순차 상태 갱신 → 로컬 통합 맵·장기 visual map 관리
```

프레임을 무조건 줄이는 대신 시각 제약이 필요한 상황을 판단하는 것이 핵심이다. 장기 visual map을 유지한다는 사실만으로 전역 루프 최적화를 수행한다고 해석하지 않는다.

## 4. 실험 결과와 해석

초록은 Hilti에서 FAST-LIVO2 대비 프레임당 실행 시간 33%, 메모리 47% 감소와 RMSE 3 cm 증가를 보고한다. x86·ARM 플랫폼 평가를 포함한다. [출처](https://arxiv.org/abs/2501.13876)

특정 데이터셋의 평균 절감률은 Vision60의 최악 지연이나 장기 메모리를 보장하지 않는다. 정확도가 약간 낮아지는 절충도 함께 고려해야 한다.

정독할 때는 비교 방법 이름뿐 아니라 센서 구성, ground truth, 궤적 정렬 방식, 실행 장치와 실패 구간 포함 여부를 함께 기록한다.

## 5. Vision60 적용 질문

다음은 논문의 검증 결과와 구분한 **프로젝트 적용 가설·검토 질문**이다.

1. 영상 선택 기준이 기하 퇴화를 놓치는 상황은 없는가?
2. 메모리 측정에 어떤 맵·버퍼·라이브러리가 포함되어 있는가?
3. FAST-LIVO2 대비 연산 절감이 제어 주기의 deadline 준수로 이어지는가?

**제안 실험:** 동일한 장거리 로그에서 두 방법의 peak memory, 처리 지연 분포, 구간별 상대 오차를 기록한다. 열·전력 제한이 있는 장치에서는 지속 실행 후에도 측정한다.

## 6. 면접형 확인 질문

각 문제는 먼저 소리 내어 답한 뒤 토글을 연다. 대학원 면접에서는 가정과 수식을, 회사 면접에서는 실패 조건과 검증 방법을 함께 말하는 연습을 한다.

### Q1. 개념·구조

퇴화 인지형 visual frame selection이 단순한 고정 frame skipping보다 나은 이유는 무엇인가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

항상 같은 비율로 버리면 LiDAR 제약이 약한 순간에도 필요한 영상을 놓칠 수 있다. 상태·환경의 정보 부족을 기준으로 영상을 선택하면 계산을 줄이면서 필요한 보완 관측을 남길 수 있다. 선택 기준 자체의 계산 비용과 오판도 포함해 평가해야 한다.

</details>

### Q2. 수학·추론

후보 영상의 information contribution을 \(\Delta H=J_V^\top R_V^{-1}J_V\)로 본다면 선택 기준을 하나 제안하고 한계를 설명하라.

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

예를 들어 \(\log\det(H+\Delta H)-\log\det(H)\), 최소 고유값 증가량, 또는 trace 기반 값을 비용 대비 비교할 수 있다. log-det은 전체 uncertainty volume, 최소 고유값은 가장 약한 방향을 본다. 선형화점·scale·모델 오류에 민감하고 계산량도 들기 때문에 근사 지표가 필요할 수 있다.

</details>

### Q3. 시스템·디버깅

평균 메모리는 줄었지만 장시간 운용 중 OOM이 발생한다. 무엇을 기록해야 하는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

RSS와 GPU memory의 peak·시간 추세, voxel·patch·keyframe 수, allocator fragmentation, queue·cache, map pruning 이벤트를 기록한다. steady-state가 존재하는지 확인하고 반복 경로와 새 영역 탐사를 분리한다. 평균 메모리 감소만으로 bounded memory를 주장할 수 없다.

</details>

### Q4. 코드 리뷰·구현

**GitHub:** [공식 또는 저자 연결 저장소](https://github.com/hku-mars/FAST-LIVO2)

이 저장소가 경량화 논문도 연결하지만, 현재 branch에 논문의 degeneration-aware frame selector와 장기 visual map이 실제 구현되어 있는지 어떻게 판별하겠는가?

<details class="study-answer" markdown="1">
<summary>답변과 채점 포인트 보기</summary>

README의 주장만으로 포함됐다고 판단하지 않는다. 논문 알고리즘의 고유 변수·지표를 기준으로 repository code search를 하고, frame 선택 조건이 호출되는 경로와 선택 결과가 buffer·map 수명에 미치는 영향을 추적한다. tag·branch·commit history와 논문 공개 시점을 확인하고, 기본 FAST-LIVO2의 `img_en`이나 단순 frame skip을 논문의 selector로 오인하지 않는다. 찾지 못하면 이 저장소를 upstream/reference implementation으로만 표기하고 paper-specific code는 미공개 또는 미확인으로 남긴다.

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

**이어 읽기:** [FAST-LIVO2 리뷰]({{ '/study/slam/state-estimation/fast-livo2/' | relative_url }}) · [전체 비교표]({{ '/study/slam/state-estimation/' | relative_url }})
