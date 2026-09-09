---
layout: study-chapter
title: "Chapter 1. 구조 — 다리 속도는 어디로 들어가나"
importance: 4
category: SLAM
series: leg_fusion
permalink: /study/slam/leg-fusion/01-architecture/
---

> **Goal:** 다리 속도가 FAST-LIO2의 **어느 단계에** 들어가는지 설명하고,
> "SLAM 두 개를 합친 게 아니라 관측을 하나 추가한 것"이라는 구조를 그림으로 전달할 수 있다.

---

# 1. 한 줄 요약

**새 SLAM을 만든 것이 아니다. 기존 FAST-LIO2 필터에 센서를 하나 더 붙인 것이다.**

```text
이런 게 아니고                          이것이다

LiDAR SLAM ─┐                          FAST-LIO2 필터 하나
            ├→ 합치기 → pose             ├─ IMU 예측
다리 SLAM  ─┘                            ├─ LiDAR 업데이트
                                         └─ 다리 속도 업데이트  ← 추가한 것
                                              ↓
                                            pose 하나
```

**최종 pose는 여전히 FAST-LIO2가 하나로 낸다.**
다리 속도는 그 안에서 추정치를 당겨주는 추가 관측일 뿐이다.

발표할 때 이 그림을 먼저 그리면 나머지가 쉽게 따라온다.

---

# 2. 전체 흐름

```text
IMU ───────────────┐
                   │
LiDAR ── scan ─────┼─→  FAST-LIO2 ESIKF  ──→  pose / odometry / TF
                   │
MCU ── body twist ─┘
   (/mcu/state/vel)
```

**순기구학은 우리가 하지 않는다.** MCU 가 관절 encoder 를 읽어 이미
몸통 강체의 6-DoF 속도로 만들어서 publish 한다. 우리는 그걸 받아 쓴다.

```text
관절 encoder ─→ [ MCU: 순기구학 + 접촉 판단 ] ─→ /mcu/state/vel
                        여기까지가 MCU 담당              ↑
                                                   우리는 여기서 시작
```

필터 내부에서는 이 순서로 돈다.

```text
① IMU 수신
      ↓
② IMU 적분으로 상태 예측          위치 · 자세 · 속도 · bias
      ↓                          여기서 불확실성 P 가 커진다
③ LiDAR 수신
      ↓
④ point-to-plane 업데이트         점과 지도 평면 비교 → 위치/자세 보정
      ↓
⑤ 다리 속도 수신
      ↓
⑥ 속도 관측 업데이트   ★          필터의 속도 vs 다리 속도
      ↓                          chi2_gate 통과하면 vel_cov 만큼 반영
⑦ odometry / TF 발행
```

**⑥만 새로 넣은 것이다.** ①~④와 ⑦은 원래 FAST-LIO2 그대로다.

---

# 3. 왜 다리 속도인가 — LiDAR가 못 보는 것

FAST-LIO2가 잘 도는 상황에서는 다리 속도를 넣어도 크게 안 바뀐다.
효과가 나오는 곳은 **LiDAR 기하가 약해지는 구간**이다.

```text
긴 복도                     벽면 법선이 전부 같은 방향
                            → 복도를 따라 미끄러지는 방향이 관측되지 않는다
                            → IMU 적분만 남고, 그건 금방 발산한다

개활지 · 특징 없는 벽        대응점이 부족

계단 착지 순간              점군이 흔들리고 IMU 가 포화될 수 있다

LiDAR 순간 끊김             프레임 몇 장이 비면 IMU 만으로 버텨야 한다
```

다리 속도는 **LiDAR와 완전히 다른 물리에서 나온다.**
관절각과 순기구학에서 나오므로 주변 환경을 전혀 안 본다.
그래서 LiDAR가 실패하는 상황과 다리가 실패하는 상황이 겹치지 않는다.

```text
LiDAR 가 약해지는 곳     특징 없는 환경, 어두움, 먼지
다리가 약해지는 곳       미끄러운 바닥, 발이 뜬 상태, 지형 변형

   → 서로를 덮어준다
```

Chapter 2의 계산에서 보겠지만, IMU 적분만으로는 **수 초 단위**로 속도가 발산한다.
다리 속도는 그 사이를 붙잡아 주는 역할이다.

---

# 4. 상태와 관측

FAST-LIO2의 상태는 대략 이렇다.

```text
x = [ 위치 p_w,  자세 R_wi,  속도 v_w,  gyro bias,  accel bias,  중력 ]
                              ↑
                     다리 속도가 건드리는 것은 여기 하나
```

관측 모델은 **필터의 속도를 다리가 보는 좌표계로 옮기는 것**이다.

$$
h(x) = R_{wi}^{\top}\, v_w
$$

```text
v_w        필터가 들고 있는 월드 좌표계 속도
R_wi       월드 ← IMU 회전
R_wi^T v_w 그 속도를 IMU 좌표계로 되돌린 것
```

다리 속도도 같은 좌표계로 맞춰야 비교가 된다.

```text
/mcu/state/vel (몸통 좌표계)
      ↓  body_to_imu_quat  회전
      ↓  body_to_imu_xyz   lever arm (Chapter 2 문제 2)
IMU 좌표계 속도
      ↓
h(x) 와 비교  →  innovation
```

## 실제 메시지

```text
타입        geometry_msgs/msg/TwistStamped
frame_id    'body'                      ← 로봇이 프레임을 직접 명시해준다
linear      x +0.20,  y −0.08,  z −0.02   m/s
angular     z −0.78 ~ +0.23               rad/s   (yaw rate)
```

세 가지를 짚고 갈 만하다.

```text
① JointState 가 아니다
     4다리 × 3관절 = 12개 값이 아니라 강체 하나의 6-DoF 속도다
     → 순기구학은 이미 끝나 있다

② frame_id 가 'body' 라고 적혀 있다
     추측할 필요가 없다. 좌표 변환의 출발점이 확정된다

③ angular 도 같이 온다   ★
     lever arm 보정에 필요한 ω 를 이 메시지에서 바로 얻을 수 있다
     (IMU gyro 를 써도 되고, 그쪽이 보통 더 고주파다)
```

**주의할 점 하나.** `TwistStamped` 에는 covariance 필드가 없다.

```text
TwistStamped                covariance 없음   ← 지금 이것
TwistWithCovarianceStamped  covariance 있음
```

그래서 `vel_cov` 를 **설정 파일의 상수로 둘 수밖에 없다.**
LIJO 처럼 속도에 따라 신뢰도를 바꾸려면 메시지 타입을 바꾸거나
우리 쪽에서 직접 계산해서 넣어야 한다.
(LIJO 리뷰 참고 — 동적 가중치가 그 논문의 핵심 기여다.)

관측으로는 `linear` 3성분만 쓰면 3 DoF 이므로 chi2 임계값이 **7.815**(95%)다.

설정에 `body_to_imu_xyz`와 `body_to_imu_quat`가 둘 다 필요한 이유가 이것이다.
**회전만으로는 부족하다.** 로봇이 회전 중이면 위치 차이가 속도 차이를 만든다.

---

# 5. 두 개의 게이트 — `chi2_gate`와 `vel_cov`

자주 헷갈리는 지점이라 발표에서 명확히 나눠주는 게 좋다.

```text
chi2_gate    반영할지 말지        yes / no
vel_cov      반영한다면 얼마나     0 ~ 100 %
```

```text
     다리 속도 도착
          ↓
   innovation 계산   y = z_leg − h(x)
          ↓
   NIS = yᵀ S⁻¹ y  가 임계값보다 큰가?
          │
     ├─ 예  → 버린다            발 미끄러짐, 센서 이상, 시간 어긋남
     │
     └─ 아니오 → vel_cov 에 따라 반영
                  작으면 강하게, 크면 약하게
```

**중요한 오해 하나를 짚고 넘어가야 한다.**

```text
✗  "차이가 작을 때만 반영한다"
✓  "차이가 비정상적으로 크면 버리고,
    정상 범위면 차이가 크든 작든 vel_cov 만큼 반영한다"
```

정상 범위 안에서는 차이가 클수록 **더 많이** 보정된다.
게이트는 이상값을 걸러내는 것이지 "작은 것만 받는" 필터가 아니다.

`max_age`도 같은 성격의 안전장치다.

```text
max_age   너무 오래된 측정은 버린다
          0.1 초 전 속도로 지금을 보정하면 오히려 나빠진다
```

---

# 6. 코드에서 건드리는 곳

`fast_lio_ros2_vision60.patch`가 하는 일:

```text
IMU_Processing.hpp     IMU 적분 과정에 상태 접근 훅 추가
laserMapping.cpp       · /mcu/state/vel 구독
                       · body → IMU 좌표 변환
                       · chi2 게이트 검사
                       · 통과하면 ESIKF 속도 업데이트 호출
```

설정 항목:

```yaml
leg_vel:
  topic: /mcu/state/vel
  vel_cov: 0.01 # (m/s)^2.  작을수록 강하게 신뢰
  chi2_gate: 7.815 # 3 DOF 95%  (Chapter 2 문제 5)
  max_age: 0.05 # 초. 이보다 오래되면 버림
  body_to_imu_xyz: [0.1, 0.0, 0.0]
  body_to_imu_quat: [0.0, 0.0, 0.0, 1.0]
```

---

# 7. 발표할 때 강조할 세 가지

```text
① 두 SLAM 을 합친 게 아니다
     기존 필터에 관측 하나를 더한 것. 출력은 여전히 하나

② 다리 속도는 "속도"만 본다
     위치나 자세를 직접 고치지 않는다.
     속도가 정확해지면 IMU 적분이 덜 발산하고,
     그 결과로 위치가 좋아지는 간접 경로다

③ 게이트와 공분산은 다른 일을 한다
     chi2_gate : 받을까 말까
     vel_cov   : 받는다면 얼마나
```

---

# 8. 예상 질문

```text
Q. LiDAR 가 잘 되는데도 넣을 이유가 있나?
A. 평소엔 거의 안 바뀐다. 복도·개활지·착지처럼
   LiDAR 기하가 약해지는 구간에서만 효과가 난다.
   그 구간을 나눠서 평가해야 효과가 보인다.

Q. 발이 미끄러지면 오히려 나빠지지 않나?
A. 그래서 chi2_gate 가 있다. 다만 "천천히 계속 미끄러지는" 경우는
   innovation 이 작아서 게이트를 통과한다. VILENS 가 이걸
   속도 bias 를 상태에 넣어서 푼다.

Q. vel_cov 는 어떻게 정하나?
A. 정지 상태에서 다리 속도의 분산을 실측해서 시작한다.
   그리고 NIS 통계로 검증한다 (Chapter 2 문제 5).

Q. 왜 위치가 아니라 속도로 넣나?
A. 다리 운동학은 상대 운동만 준다. 절대 위치 정보가 없다.
   위치로 넣으려면 적분해야 하고, 그러면 drift 가 따라온다.
```

---

# 9. Chapter 연결

```text
Chapter 1  ← 여기
구조 — 어디에 들어가나

Chapter 2
계산 예제 — 좌표 변환, 칼만 갱신, chi2 gate 를 숫자로

SLAM History Chapter 2
EKF, Kalman gain, 공분산 — 여기 쓰인 수학의 출처

VILENS 리뷰
다리 속도 bias 를 상태로 추정하는 방식

LIJO 리뷰
관절 속도 + 속도 의존 동적 가중치
```
