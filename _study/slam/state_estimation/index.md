---
layout: study-series
title: "상태 추정 계보 — LiDAR에서 LIVO·다리 센서 융합까지"
description: "FAST-LIO2 기준 논문 비교: 센서, 추정 구조, 루프클로저, Vision60 적용 관점."
category: SLAM
series: state_estimation
series_index: true
importance: 1
permalink: /study/slam/lio/
redirect_from:
  - /study/slam/state-estimation/
---

현재 Vision60에 적용된 FAST-LIO2의 원리와 한계를 이해하고, 단차·슬립·충격에서 어떤 정보를 추가할지 판단하기 위한 첫 번째 공부 페이지다. 핵심 비교 대상은 FAST-LIO2, FAST-LIVO2, VILENS, LIO-SAM, LTA-OM, Point-LIO, GLIM이다. 계보를 설명하는 선행 연구와 최근 후속 연구를 함께 정리한다.

**자료 확인일: 2026-09-07.** 프로젝트와 관련된 대표 연구를 선별한 목록이며, 최신 논문 전체를 망라한 순위표는 아니다. 논문의 구조·기여와 Vision60에 대한 적용 가설을 구분한다. 센서 구성, 연산 장치, 데이터셋이 다른 논문의 정확도·실행 속도를 하나의 숫자로 직접 비교하지 않는다.

## 1. 핵심 논문 비교

L은 LiDAR, I는 IMU, V는 카메라, K는 관절·다리 운동학 정보를 뜻한다. **루프클로저**는 과거 장소를 다시 관측해 누적 오차를 보정하는 기능이다. 아래의 포함 여부는 논문이 다루는 시스템 기준이며, 별도 확장 패키지는 구분한다. 표가 화면보다 넓으면 가로로 스크롤할 수 있다.

<div class="table-responsive" markdown="1">

| 논문 · 발표 | 센서 / 추정 구조 | 핵심 기여 3개 | 맵·루프클로저·백엔드 | Vision60에서 읽을 이유 / 확인할 한계 |
|:---|:---|:---|:---|:---|
| **[FAST-LIO2]({{ '/study/slam/state-estimation/fast-lio2/' | relative_url }})** · T-RO 2022 | L + I / tightly coupled iterated EKF | ① 원시 점의 direct scan-to-map 등록<br>② 효율적인 반복 필터 갱신<br>③ ikd-Tree의 증분 삽입·삭제·재균형 | 로컬 점군 맵 생성. 원 논문에는 전역 루프클로저·포즈 그래프 보정 없음 | **현재 기준선.** IMU 전파 → deskew → 점-평면 잔차 → 상태·맵 갱신을 추적. 기하 퇴화와 누적 drift를 구분해 측정 |
| **[LIO-SAM]({{ '/study/slam/state-estimation/lio-sam/' | relative_url }})** · IROS 2020 | L + I, 선택적 GPS / factor graph + iSAM2 | ① IMU 사전적분과 deskew<br>② 특징 기반 로컬 scan matching<br>③ LiDAR·GPS·루프 제약의 그래프 통합 | 키프레임 기반 매핑과 루프 제약 포함. [공식 구현](https://github.com/TixiaoShan/LIO-SAM)은 IMU 추정 그래프와 매핑 그래프를 구분 | **필터와 그래프의 대조군.** 루프를 붙이기 쉬운 구조가 오검출까지 해결하는 것은 아님. 공식 루프 구현은 proof of concept |
| **[Point-LIO]({{ '/study/slam/state-estimation/point-lio/' | relative_url }})** · Advanced Intelligent Systems 2023 | L + I / point-by-point 필터 | ① 점의 측정 시각마다 상태 갱신<br>② IMU를 관측으로 사용하는 운동 모델<br>③ 고대역폭·격렬한 운동 대응 | 오도메트리·매핑 중심. 전역 루프 백엔드는 별도 | **충격·빠른 회전 비교군.** 프레임 누적에 따른 왜곡과 IMU 측정 범위 문제를 읽기. 높은 출력률을 낮은 지연·정확도와 동일시하지 않기 |
| **[FAST-LIVO2]({{ '/study/slam/state-estimation/fast-livo2/' | relative_url }})** · T-RO 2025 | L + I + V / ESIKF 순차 갱신 | ① LiDAR 기하·영상 광도 잔차의 순차 융합<br>② 점과 이미지 패치를 연결하는 통합 voxel map<br>③ 평면 사전정보·노출 추정으로 영상 정합 보강 | LiDAR와 영상이 같은 맵을 사용. 원 논문은 전역 루프 보정보다 odometry에 초점 | **카메라 추가 후보.** LiDAR 기하가 약할 때 영상이 보완할 조건을 확인. 어두움·블러·무텍스처 및 시간 동기화·외부 파라미터를 함께 검토 |
| **[VILENS]({{ '/study/slam/state-estimation/vilens/' | relative_url }})** · T-RO, 온라인 2022 / 권호 2023 | V + I + L + K / factor graph | ① 네 센서 모달리티의 tightly coupled 융합<br>② 다리 속도의 사전적분 factor<br>③ 온라인 선속도 bias 추정 | 다중 센서 odometry가 중심. factor graph 사용 자체가 장소인식·루프클로저 포함을 뜻하지 않음 | **슬립·다리 정보의 핵심 비교군.** ANYmal의 험지 실험과 속도 bias의 의미를 읽기. Vision60의 관절·접촉 정보 접근성과 운동학 모델 확인 필요 |
| **[LTA-OM]({{ '/study/slam/state-estimation/lta-om/' | relative_url }})** · JFR 2024 | L + I / FAST-LIO2 기반 LIO + 전역 보정 | ① STD 기반 루프 검출·보정<br>② false-positive 루프 기각<br>③ 보정된 과거 맵을 활용하는 장기 연관 | 루프클로저, 장기 매핑, 다중 세션 localization·mapping 포함. 과거 맵의 보정 결과가 LIO 등록에 다시 사용됨 | **모듈화 목표의 비교군.** 프론트엔드·루프 검출·전역 보정의 경계와 보정 맵의 재사용 경로를 읽기 |
| **[GLIM]({{ '/study/slam/state-estimation/glim/' | relative_url }})** · Robotics and Autonomous Systems 2024 | L + I, 다중 카메라 제약 지원 / fixed-lag smoothing + 전역 최적화 | ① GPU 가속 scan matching factor<br>② 시간 창 내 상태의 공동 최적화<br>③ 서브맵 간 등록 오차의 전역 최소화 | 로컬 odometry와 전역 서브맵 최적화가 연결된 매핑 시스템 | **최적화 기반 강건성 비교군.** 일시적인 기하 퇴화에서 과거 상태를 활용하는 방식과 GPU·메모리 부담을 확인 |

</div>

FAST-LIO2의 **direct**는 기하 잔차나 대응점 계산을 생략한다는 뜻이 아니다. 사전에 모서리·평면 특징점을 추출하는 단계를 거치지 않고 원시 점을 맵에 등록한다는 의미다. FAST-LIVO2의 영상 direct 방식은 특징점 기술자 매칭 대신 광도 오차를 사용한다. 두 방식에서 무엇을 관측 잔차로 삼는지 구분해 읽는다. [FAST-LIO2 원문](https://arxiv.org/abs/2107.06829), [FAST-LIVO2 원문](https://arxiv.org/abs/2408.14035)

필터 계열은 현재 상태와 불확실성을 갱신하는 데 집중하고, smoothing 계열은 여러 시점의 상태를 함께 최적화한다. 그러나 **필터 / 그래프**와 **오도메트리 / 루프클로저 포함 SLAM**은 서로 다른 비교 축이다. FAST-LIO2에 전역 백엔드를 연결할 수 있고, VILENS처럼 그래프를 사용하는 odometry도 있다. LTA-OM은 필터 기반 프론트엔드와 전역 보정을 연결한 사례다. [VILENS 원문](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf), [LTA-OM 원문](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.22337)

## 2. 계보를 이해하는 선행 연구

<div class="table-responsive" markdown="1">

| 논문 | 계보에서의 위치 | 다음 논문과 비교할 점 |
|:---|:---|:---|
| [LOAM: Lidar Odometry and Mapping in Real-time]({{ '/study/slam/state-estimation/loam/' | relative_url }}) · RSS 2014 | 고주파 odometry와 저주파 정밀 mapping을 분리한 LiDAR 중심 출발점. IMU 보조 사용 가능 | 특징 기반 정합과 두 단계 처리 구조를 LIO-SAM·FAST-LIO2와 비교 |
| [FAST-LIO]({{ '/study/slam/state-estimation/fast-lio/' | relative_url }}) · 프리프린트 2020 | LiDAR 특징점과 IMU를 반복 EKF로 강결합. 상태 차원에 기반한 효율적인 Kalman gain 계산 | FAST-LIO2가 필터 기반을 유지하면서 direct 등록·맵 자료구조를 바꾸는 이유 |
| [FAST-LIVO]({{ '/study/slam/state-estimation/fast-livo/' | relative_url }}) · IROS 2022 | LiDAR·관성·영상의 sparse-direct 융합 | FAST-LIVO2의 통합 맵·순차 갱신·영상 정합 개선이 해결하는 문제 |

</div>

학습 흐름은 **LiDAR 정합 → 관성과 강결합한 LIO → 영상까지 결합한 LIVO → 다리 정보를 사용하는 상태 추정**으로 잡는다. 이것이 모든 방법의 직접적인 후속 관계나 우열을 뜻하지는 않는다. LIO 안에서도 LIO-SAM의 그래프와 FAST-LIO의 필터로 설계가 갈리고, VILENS는 네 모달리티를 결합하는 별도의 분기다. 루프클로저와 장기 매핑은 센서 수 증가와 별개로 살펴본다.

## 3. 최근 확장 연구 — 2025~2026

아래 연구는 핵심 7편을 읽은 뒤 확장할 후보다. 최신이라는 이유만으로 Vision60 적용 우선순위를 높이지 않는다. 특히 LIJO의 비교는 출판사 초록에 근거한 1차 정리이며, 정독 시 실험 설정·ablation·구현 공개 여부를 추가 확인해야 한다.

<div class="table-responsive" markdown="1">

| 논문 · 자료 | 입력 / 접근법 | 새롭게 읽을 핵심 3개 | 프로젝트 연결 및 남은 검증 |
|:---|:---|:---|:---|
| [FAST-LIVO2 on Resource-Constrained Platforms: LiDAR-Inertial-Visual Odometry with Efficient Memory and Computation]({{ '/study/slam/state-estimation/fast-livo2-resource-constrained/' | relative_url }}) · 2025 | L + I + V / FAST-LIVO2 경량화 | ① 퇴화를 고려한 영상 프레임 선택<br>② 로컬 통합 맵과 장기 visual map<br>③ 계산·메모리 절감 | 온보드 자원이 제한될 때 후보. 논문의 x86·ARM 결과를 실제 탑재 장치에서 재검증 |
| [Legolas: Deep Leg-Inertial Odometry]({{ '/study/slam/state-estimation/legolas/' | relative_url }}) · PMLR 2025, 제8회 CoRL 논문집 | I + 다리 센서 / 학습 기반 odometry | ① 다리·관성 신호에서 odometry 학습<br>② 실세계 학습 궤적 수집에 의존하지 않는 접근<br>③ 두 실제 4족 플랫폼 평가 | 외부 센서가 어려운 상황의 보조 추정 후보. Vision60의 보행·마찰·센서 분포에 대한 일반화는 별도 검증. LiDAR 매핑 시스템과 구분 |
| [Smooth LiDAR–Inertial–Joint Odometry for perception-driven legged locomotion (LIJO)]({{ '/study/slam/state-estimation/lijo/' | relative_url }}) · Robot Learning, 2026-08-10 출판 | L + I + 관절 encoder / manifold EKF | ① 운동학 기반 몸통 속도 제약<br>② 운동 속도에 따른 동적 가중치<br>③ IMU를 관측으로 사용해 odometry jitter 억제 | 관절 융합과 제어 입력의 부드러움을 함께 보는 최신 후보. 속도 기반 가중치가 접촉 상태·슬립을 얼마나 구별하는지와 지연을 정독 시 확인 |

</div>

## 4. 표를 Vision60 문제에 연결해서 읽기

다음은 논문 성능을 확정하는 결론이 아니라 **실험으로 확인할 적용 가설**이다. 먼저 같은 로그에서 FAST-LIO2의 실패 시점과 원인을 특정해야 한다. 점군 왜곡, 높이 drift, 속도 jitter는 다른 현상이며 하나의 보정으로 모두 해결된다고 가정하지 않는다.

<div class="table-responsive" markdown="1">

| 관찰할 현상 | 비교해서 읽을 논문 | 먼저 확인할 데이터 / 판단 기준 |
|:---|:---|:---|
| 착지 충격·빠른 회전에서 점군이 휘거나 pose가 튐 | FAST-LIO2 → Point-LIO | 점별 timestamp, IMU 포화·누락, 시간 동기화, deskew 결과. 출력률과 end-to-end 지연을 따로 측정 |
| 벽·복도·개활지에서 특정 방향 drift | FAST-LIO2 → FAST-LIVO2·GLIM | 기하 제약이 약한 방향, 영상 텍스처·블러, 구간별 상대 오차. 영상 추가 효과와 연산 비용을 함께 비교 |
| 슬립·단차에서 속도 또는 높이 오차 증가 | VILENS → LIJO, 보조로 Legolas | 관절각·관절속도·접촉 추정·IMU의 동기화, 운동학 속도와 외부 추정의 불일치. 슬립 구간과 비슬립 구간을 나눠 평가 |
| 정지·저속에서 pose jitter가 제어 입력에 전달됨 | FAST-LIO2 → LIJO | 정지 구간 pose·속도 분산과 주파수 성분, 필터링 후 지연. 부드러운 출력이 실제 운동을 지우지 않는지 확인 |
| 장거리 이동 후 재방문 시 맵이 겹치지 않음 | LIO-SAM → LTA-OM·GLIM | 루프 전후 궤적·맵 일관성, 오검출 여부, 전역 보정 시 pose 불연속. 재방문 보정과 순간 추정 실패를 구분 |

</div>

프론트엔드는 센서 입력에서 연속적인 pose·속도와 정합용 맵을 만들고, 루프 검출은 과거 장소 후보와 상대 변환을 제안하며, 백엔드는 채택한 제약으로 전역 궤적·맵을 보정한다. 이는 비교를 위한 기능 구분이다. 실제 구현에서는 LTA-OM처럼 보정된 맵이 프론트엔드로 돌아가므로, 모듈 이름뿐 아니라 **어떤 좌표계의 상태·점군·제약이 어느 시점에 전달되는지**까지 확인해야 한다. [LTA-OM 원문](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.22337)

이번 공부에서는 **FAST-LIO2 → LIO-SAM → Point-LIO → FAST-LIVO2 → VILENS → LTA-OM → GLIM** 순서로 읽으면 현재 시스템에서 설계 차이를 넓혀 갈 수 있다. 논문별 리뷰는 ① 상태와 관측 잔차, ② 해결하는 실패 모드와 실험 근거, ③ Vision60에 추가할 입력·인터페이스의 세 항목으로 정리한다. 최신 확장은 관측된 문제가 연산 자원, 슬립, jitter 중 어디에 가까운지에 따라 선택한다.

## 5. 서지 정보와 해석 주의점

FAST-LIO2는 프리프린트가 2021년, T-RO 게재가 2022년이다. FAST-LIVO2는 온라인 출판이 2024-11-19이고 T-RO 41권의 2025년 논문으로 인용된다. VILENS는 2022년 온라인 출판과 2023년 39권 1호를 구분하면 된다. [FAST-LIO2]({{ '/study/slam/state-estimation/fast-lio2/' | relative_url }}), [FAST-LIVO2 출판 정보](https://ieeexplore.ieee.org/document/10757429/), [VILENS 출판 정보](https://ora.ox.ac.uk/objects/uuid%3A031d8565-2b83-4ee0-a17f-006471c7f185)

GLIM 자체의 논문은 **2024년**이다. 저자 저장소에서 연결한 ICRA 2022 논문은 GLIM의 기반 연구이므로 구분한다. 또한 FAST-LIVO2를 유일한 최신 표준으로 단정하거나 FAST-LIO2를 “백엔드를 포기한 방법”으로만 요약하기보다, 각 논문이 다루는 추정 범위와 실제 추가 가능한 모듈을 비교한다. [GLIM 저자 저장소](https://github.com/koide3/glim)
