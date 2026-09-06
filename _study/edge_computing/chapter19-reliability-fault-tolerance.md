---
title: "Chapter 19. Reliability & Fault Tolerance"
importance: 20
---

> **Goal:** Robot system에서 failure가 언젠가는 발생한다고 가정하고,
> 문제가 생겼을 때 얼마나 안전하게 버티고, 복구하고, 계속 동작할 수 있는지 이해한다.
>
> Fault, Error, Failure, Redundancy, Watchdog, Retry, Timeout,
> Graceful Degradation, Fail-Safe, Fail-Operational,
> Single Point of Failure, Recovery, Redundant Sensor의 개념을
> 실제 robot architecture와 연결하는 것이 목표다.

---

# 1. Reliability란?

Reliability는:

> 시스템이 일정 기간 동안 요구된 기능을 정상적으로 수행할 가능성

과 관련된 개념이다.

쉽게 말하면:

```text
얼마나 잘 안 고장나는가?
```

이다.

하지만 실제 robot system에서는:

```text
고장이 절대 안 남
```

을 기대하기 어렵다.

그래서 더 중요한 질문은:

```text
고장이 나면 어떻게 되는가?
```

이다.

---

# 2. Fault Tolerance

Fault Tolerance는:

> 일부 component에 문제가 생겨도 전체 system이 계속 기능하거나
> 안전하게 동작할 수 있도록 만드는 능력

이다.

예:

```text
Camera 1 failure
     ↓
Camera 2 available
     ↓
Perception continues
```

이런 구조가 fault tolerant하다.

---

# 3. Fault, Error, Failure

세 단어는 비슷하지만 구분하면 좋다.

```text
Fault
→ 문제의 원인

Error
→ 내부 상태가 잘못된 상태

Failure
→ 외부에서 요구 기능을 수행하지 못한 결과
```

---

# 4. 예시

LiDAR cable이 빠졌다고 하자.

```text
Fault
→ Cable disconnected
```

그 결과:

```text
Error
→ No LiDAR data
```

그리고:

```text
Failure
→ Localization unavailable
```

가 될 수 있다.

---

# 5. 또 다른 예

SSD가 망가졌다고 하자.

```text
Fault
→ Storage hardware fault

Error
→ Write operation fails

Failure
→ rosbag recording unavailable
```

이다.

---

# 6. Reliability와 Fault Tolerance는 다르다

```text
Reliability
→ 고장이 얼마나 적게 발생하는가?

Fault Tolerance
→ 고장이 발생해도 얼마나 버틸 수 있는가?
```

이다.

둘 다 중요하다.

---

# 7. Robot은 복잡한 System이다

Vision60 같은 robot은:

```text
Battery
Motors
MCU
Jetson
LiDAR
IMU
Camera
Network
Storage
ROS 2
SLAM
Navigation
```

많은 component로 이루어진다.

Component 수가 늘어날수록 failure point도 많아진다.

---

# 8. Single Point of Failure

어떤 하나의 component failure가 전체 system failure로 이어지는 경우:

```text
Single Point of Failure
```

라고 한다.

예:

```text
Single Power Supply
       ↓ failure
Entire Compute Down
```

---

# 9. SPOF

Single Point of Failure를 줄여:

```text
SPOF
```

라고도 한다.

시스템 설계에서는:

```text
Where is the SPOF?
```

를 찾는 것이 중요하다.

---

# 10. 예: LiDAR 하나만 사용하는 SLAM

```text
LiDAR
   ↓
SLAM
   ↓
Localization
```

LiDAR가 죽으면:

```text
Localization lost
```

가 된다.

이 경우 LiDAR는 중요한 SPOF일 수 있다.

---

# 11. Redundancy

같은 기능을 수행할 수 있는 component를 여러 개 두는 것을:

```text
Redundancy
```

라고 한다.

예:

```text
Sensor A
Sensor B
```

한 sensor가 실패해도 다른 sensor를 사용할 수 있다.

---

# 12. Hardware Redundancy

예:

```text
Dual IMU
Dual Network Link
Dual Power Supply
```

같은 구조가 hardware redundancy다.

---

# 13. Software Redundancy

Software에서도 redundancy를 사용할 수 있다.

예:

```text
Primary localization
Backup localization
```

또는:

```text
Algorithm A
Algorithm B
```

결과를 비교할 수 있다.

---

# 14. Redundancy가 항상 좋은 것은 아니다

장점:

```text
Reliability ↑
Fault tolerance ↑
```

하지만:

```text
Cost ↑
Weight ↑
Power ↑
Complexity ↑
Maintenance ↑
```

도 증가한다.

Robot에서는 특히 weight와 power가 중요하다.

---

# 15. Active Redundancy

두 component를 동시에 사용한다.

예:

```text
IMU A
IMU B
```

둘 다 계속 측정한다.

한쪽 문제가 발생하면 즉시 다른 쪽을 사용할 수 있다.

---

# 16. Standby Redundancy

평소에는 primary만 사용한다.

```text
Primary
active

Backup
standby
```

Primary failure 시 backup을 시작한다.

---

# 17. Hot Standby

Backup이 이미 실행 중이어서 빠르게 전환 가능:

```text
Primary
ACTIVE

Backup
READY
```

---

# 18. Cold Standby

Backup은 평소 꺼져 있다.

Failure 후:

```text
Start Backup
```

하므로 복구 시간이 더 길 수 있다.

---

# 19. Fault Detection

Fault tolerance를 하려면 먼저:

```text
문제가 생겼다는 것을 알아야 한다.
```

즉:

```text
Fault Detection
```

이 필요하다.

---

# 20. Detection Example

LiDAR:

```text
Expected:
10 Hz

Actual:
0 Hz
```

이면 fault를 감지할 수 있다.

---

# 21. Fault Isolation

문제가 있다는 것만으로는 부족하다.

어디가 문제인지 좁히는 과정:

```text
Fault Isolation
```

이다.

예:

```text
LiDAR hardware?
Network?
Driver?
ROS?
FAST-LIO2?
```

---

# 22. Fault Identification

좀 더 구체적으로:

```text
What exactly failed?
```

를 판단한다.

예:

```text
Ethernet cable disconnected
```

처럼 원인을 특정한다.

---

# 23. FDI

이 세 개를 묶어:

```text
Fault Detection and Isolation
```

또는 넓게:

```text
Fault Detection and Identification
```

이라고 부르는 경우가 있다.

---

# 24. Health Monitoring

Chapter 18에서 배운 observability가 fault detection의 기반이 된다.

예:

```text
LiDAR Hz
IMU Hz
CPU Temp
Disk Usage
Process Heartbeat
```

을 계속 확인한다.

---

# 25. Heartbeat

Component가 살아 있음을 주기적으로 알리는 signal:

```text
Heartbeat
```

이다.

예:

```text
FAST-LIO2
    │
    │ 10 Hz heartbeat
    ▼
Health Monitor
```

---

# 26. Heartbeat Timeout

예:

```text
Expected:
heartbeat every 100 ms

No heartbeat for:
500 ms
```

이면:

```text
Component failure suspected
```

로 판단할 수 있다.

---

# 27. Timeout

Timeout은:

> 일정 시간 동안 응답이 없으면 더 이상 기다리지 않고 failure로 판단

하는 mechanism이다.

---

# 28. Network Timeout

예:

```text
Send Command
   ↓
Wait for Response
   ↓
No response for 1 s
   ↓
Timeout
```

---

# 29. 왜 Timeout이 필요한가?

Timeout이 없으면:

```text
Wait forever
```

가 될 수 있다.

하나의 component failure가 전체 pipeline을 멈추게 할 수 있다.

---

# 30. Timeout 값을 너무 짧게 잡으면?

정상적인 network jitter에도 failure로 판단할 수 있다.

```text
False Positive
```

가 증가한다.

---

# 31. Timeout을 너무 길게 잡으면?

실제 failure를 늦게 감지한다.

```text
Failure Detection Delay ↑
```

따라서 실제 timing requirement를 기반으로 설정해야 한다.

---

# 32. Retry

작업이 실패했을 때 다시 시도하는 것:

```text
Retry
```

이다.

예:

```text
Cloud upload failed
      ↓
Retry
```

---

# 33. Retry가 유용한 경우

일시적인:

```text
Network glitch
Temporary server error
Packet loss
```

같은 문제에는 retry가 도움이 된다.

---

# 34. Retry가 위험한 경우

근본적으로 실패한 상황:

```text
Wrong credential
Disk full
Hardware disconnected
```

에서 계속 retry하면:

```text
CPU waste
Network flooding
Log flooding
```

이 발생할 수 있다.

---

# 35. Retry Limit

예:

```text
Retry 3 times
```

후:

```text
Declare failure
```

처럼 제한할 수 있다.

---

# 36. Exponential Backoff

Chapter 17에서 배운 것처럼:

```text
1 s
2 s
4 s
8 s
16 s
```

처럼 retry interval을 늘릴 수 있다.

Server/network에 부담을 줄인다.

---

# 37. Jittered Backoff

Robot 100대가 동시에 reconnect하면 모두:

```text
1 s
2 s
4 s
```

에 동시에 요청할 수 있다.

그래서 약간의 random delay를 추가할 수 있다.

예:

```text
4 s ± random
```

---

# 38. Watchdog

Watchdog은:

> System/component가 멈췄는지 감시하고 필요 시 recovery action을 수행

하는 mechanism이다.

---

# 39. Software Watchdog

예:

```text
FAST-LIO2
   │ heartbeat
   ▼
Watchdog Process
```

heartbeat가 끊기면:

```text
Restart FAST-LIO2
```

할 수 있다.

---

# 40. Hardware Watchdog

SoC/MCU에 hardware timer가 있을 수 있다.

Software가 주기적으로 watchdog을 reset해야 한다.

```text
Software healthy
→ Feed watchdog
```

Software가 멈추면:

```text
No feed
   ↓
Timeout
   ↓
System Reset
```

---

# 41. Watchdog Feed

주기적으로 watchdog timer를 reset하는 것을 흔히:

```text
Feed the watchdog
```

또는:

```text
Kick the watchdog
```

이라고 표현한다.

---

# 42. Watchdog의 문제

잘못 설계하면:

```text
Crash
Restart
Crash
Restart
```

무한 restart loop가 생길 수 있다.

---

# 43. Restart Storm

여러 service가 서로 dependency가 있는데
계속 재시작하면 system 전체가 불안정해질 수 있다.

이를 흔히:

```text
Restart Storm
```

이라고 부를 수 있다.

---

# 44. Restart Backoff

예:

```text
1st restart → immediate
2nd → 5 s
3rd → 30 s
4th → stop and alert
```

같은 정책을 사용할 수 있다.

---

# 45. Recovery

Failure 후 정상 상태로 돌아오는 과정:

```text
Recovery
```

이다.

---

# 46. Recovery Level

Recovery는 여러 단계로 할 수 있다.

```text
Level 1
Retry operation

Level 2
Restart node

Level 3
Restart container

Level 4
Restart service stack

Level 5
Reboot computer

Level 6
Operator intervention
```

---

# 47. 가장 작은 Recovery부터

가능하다면:

```text
Restart entire robot
```

보다:

```text
Restart failed component only
```

가 좋다.

왜냐하면 영향 범위가 작기 때문이다.

---

# 48. Blast Radius

Failure나 recovery가 영향을 미치는 범위를:

```text
Blast Radius
```

라고 표현할 수 있다.

예:

```text
Restart camera node
→ small blast radius

Reboot Jetson
→ large blast radius
```

---

# 49. Isolation

한 component failure가 다른 component로 퍼지지 않게 하는 것:

```text
Fault Isolation
```

과 연결된다.

Docker/container 분리도 일부 도움이 될 수 있다.

---

# 50. Process Isolation

예:

```text
Camera process crashes
```

해도:

```text
FAST-LIO2
Navigation
```

까지 같이 죽지 않도록 process를 분리할 수 있다.

---

# 51. Container Isolation

예:

```text
Perception Container
SLAM Container
Fleet Agent Container
```

를 나누면 dependency/failure 범위를 줄일 수 있다.

하지만 container가 완벽한 fault isolation을 보장하는 것은 아니다.

---

# 52. Resource Isolation

한 process가 모든 RAM을 사용하면
다른 process까지 죽을 수 있다.

그래서:

```text
CPU Limit
Memory Limit
Queue Limit
Disk Quota
```

같은 resource isolation을 고려할 수 있다.

---

# 53. Memory Leak

Program이 memory를 계속 할당하고 해제하지 않으면:

```text
Memory Leak
```

이 생긴다.

```text
1 GB
2 GB
5 GB
10 GB
...
```

결국 OOM으로 이어질 수 있다.

---

# 54. Memory Limit

Container/process에 memory limit을 두면
system 전체를 보호하는 데 도움이 될 수 있다.

하지만 memory가 부족해 해당 component가 먼저 죽을 수 있으므로
recovery 설계와 함께 봐야 한다.

---

# 55. Disk Full Fault

Chapter 13에서 배운 것처럼 disk full은 흔한 failure다.

```text
rosbag
Docker log
System log
```

가 계속 쌓이면 disk가 가득 찬다.

---

# 56. Disk Full을 Fault Tolerant하게 처리

예:

```text
Disk > 80%
→ Warning

Disk > 90%
→ Stop optional recording

Disk > 95%
→ Preserve critical logs only
```

처럼 단계적으로 degrade할 수 있다.

---

# 57. Graceful Degradation

System 전체를 바로 멈추는 대신
일부 기능을 낮춰서 계속 동작하는 것:

```text
Graceful Degradation
```

이다.

---

# 58. Example: Camera Failure

평상시:

```text
LiDAR
Camera
IMU
```

모두 사용.

Camera failure:

```text
LiDAR + IMU SLAM continues
Vision features disabled
```

이런 동작이 graceful degradation이다.

---

# 59. Example: Cloud Failure

```text
Internet Lost
```

되어도:

```text
Local SLAM
Navigation
Safety
```

는 계속 동작.

```text
Telemetry Upload
OTA
```

만 중단.

---

# 60. Example: Disk Almost Full

```text
Raw camera recording OFF
Critical logs ON
SLAM ON
```

처럼 중요한 기능을 우선한다.

---

# 61. Degraded Mode

기능이 일부 제한된 상태를:

```text
Degraded Mode
```

라고 정의할 수 있다.

예:

```text
NORMAL
DEGRADED
SAFE_STOP
ERROR
```

---

# 62. State Machine

Reliability system은 state machine으로 관리하기 좋다.

```text
BOOT
 ↓
NORMAL
 ↓ fault
DEGRADED
 ↓ severe fault
SAFE_STOP
```

Recovery:

```text
DEGRADED
 ↓ recovered
NORMAL
```

---

# 63. Fail-Safe

Failure가 발생하면:

> 안전한 상태로 가는 것

을:

```text
Fail-Safe
```

라고 한다.

---

# 64. Example

Localization이 사라졌는데 robot이 계속 자율주행하면 위험할 수 있다.

```text
Localization Lost
      ↓
Stop Autonomous Motion
```

이 fail-safe 동작일 수 있다.

---

# 65. Fail-Operational

Failure가 생겨도:

> 필요한 기능을 계속 수행

하는 것을:

```text
Fail-Operational
```

이라고 한다.

---

# 66. Fail-Safe vs Fail-Operational

```text
Fail-Safe
→ 문제가 생기면 안전하게 정지

Fail-Operational
→ 문제가 생겨도 기능을 계속 수행
```

이다.

---

# 67. 어떤 것을 선택할까?

Application에 따라 다르다.

예:

```text
Construction Robot
Localization lost
```

라면 안전 정지가 더 적절할 수 있다.

하지만:

```text
Aircraft flight control
```

처럼 즉시 멈출 수 없는 system은 fail-operational 요구가 더 강할 수 있다.

---

# 68. Safety와 Availability Trade-off

계속 동작하는 것이 항상 좋은 것은 아니다.

```text
Availability ↑
```

를 위해 불확실한 sensor를 계속 사용하면
safety가 떨어질 수 있다.

---

# 69. Confidence-Based Degradation

예:

```text
Localization confidence high
→ Normal

Confidence medium
→ Slow speed

Confidence low
→ Stop
```

같은 방식도 가능하다.

---

# 70. Redundant Sensors

예:

```text
LiDAR
Camera
IMU
Joint Encoder
```

가 서로 다른 방식으로 motion/environment를 측정한다.

이런 heterogeneous redundancy는 유용하다.

---

# 71. Homogeneous Redundancy

같은 종류 sensor 여러 개:

```text
IMU A
IMU B
IMU C
```

---

# 72. Heterogeneous Redundancy

다른 종류 sensor로 같은 state를 보완:

```text
LiDAR
Camera
Leg Odometry
```

모두 robot motion에 대한 정보를 제공할 수 있다.

---

# 73. Common-Mode Failure

Redundant sensor를 두 개 달아도
둘이 같은 원인으로 동시에 죽을 수 있다.

예:

```text
Two Cameras
     │
     └── Same power supply
              ↓ failure
       Both cameras lost
```

이를:

```text
Common-Mode Failure
```

라고 한다.

---

# 74. Common-Cause Failure

하나의 공통 원인이 여러 component를 동시에 실패시키는 것을:

```text
Common-Cause Failure
```

라고 한다.

예:

```text
Overheat
Power loss
Water ingress
Network switch failure
```

---

# 75. Redundancy 설계 시 중요한 것

단순히:

```text
2개 달기
```

가 아니라:

```text
Independent Power?
Independent Network?
Different Failure Mode?
```

까지 생각해야 한다.

---

# 76. Sensor Voting

3개의 sensor가 있다고 하자.

```text
Sensor A = 10.0
Sensor B = 10.1
Sensor C = 50.0
```

C가 이상하다고 판단할 수 있다.

이를 voting/consistency checking으로 처리할 수 있다.

---

# 77. Majority Voting

3개 component 중 2개가 같은 결과를 내면
그 결과를 선택하는 방식.

```text
A → 1
B → 1
C → 0

Result → 1
```

---

# 78. Triple Modular Redundancy

세 개의 동일한 subsystem과 voter를 사용하는 구조를:

```text
TMR
=
Triple Modular Redundancy
```

라고 한다.

Safety-critical system에서 볼 수 있다.

---

# 79. Robot에서는 항상 TMR을 쓰나?

아니다.

비용과 무게가 매우 크다.

Robot에서는 필요한 critical subsystem에 선택적으로 redundancy를 적용한다.

---

# 80. Plausibility Check

Sensor 값이 물리적으로 가능한 범위인지 확인할 수 있다.

예:

```text
Robot speed:
500 m/s
```

이면 명백히 이상하다.

---

# 81. Range Check

예:

```text
Battery voltage
Temperature
Joint angle
```

가 allowed range 밖이면 fault로 볼 수 있다.

---

# 82. Rate-of-Change Check

값 자체는 정상 범위인데
갑자기 너무 빠르게 바뀌는 것도 이상할 수 있다.

예:

```text
Pose:
0 m → 100 m in 10 ms
```

---

# 83. Cross-Sensor Consistency

예:

```text
IMU says robot stationary

Leg odometry says 5 m/s
```

이면 sensor 중 하나가 이상할 가능성이 있다.

---

# 84. Residual Monitoring

State estimator에서:

```text
Prediction
vs
Measurement
```

차이를 residual로 볼 수 있다.

Residual이 갑자기 매우 커지면 sensor fault나 model 문제를 의심할 수 있다.

---

# 85. Covariance와 Health

Estimator의 covariance가 급격히 커지면:

```text
State uncertainty ↑
```

를 의미할 수 있다.

이를 degraded mode trigger로 사용할 수 있다.

---

# 86. Localization Health Example

```text
Position covariance small
→ Healthy

Covariance increasing
→ Warning

Localization invalid
→ Safe Stop
```

처럼 상태를 나눌 수 있다.

---

# 87. Fault Injection

Reliability를 테스트하려면 실제 failure를 일부러 만들어 볼 수 있다.

이를:

```text
Fault Injection
```

이라고 한다.

---

# 88. Example Fault Injection

```text
LiDAR cable disconnect
Camera disable
Network disconnect
Disk fill
Kill FAST-LIO2 process
CPU overload
```

등을 일부러 발생시킨다.

---

# 89. 왜 Fault Injection을 할까?

정상 상황만 테스트하면 recovery path를 검증할 수 없다.

질문:

```text
LiDAR가 실제로 죽으면 robot은 어떻게 행동하는가?
```

를 실제로 확인해야 한다.

---

# 90. Chaos Engineering

Distributed/cloud system에서는
의도적으로 failure를 주입해 resilience를 검증하는 접근을:

```text
Chaos Engineering
```

이라고 한다.

Robot에도 일부 개념을 적용할 수 있지만
physical safety를 매우 신중히 고려해야 한다.

---

# 91. Robot Fault Test는 안전하게

실제 robot에서 fault injection을 할 때는:

```text
Robot secured?
Low speed?
E-stop available?
Human nearby?
Safe test area?
```

를 먼저 확인해야 한다.

---

# 92. FMEA

Reliability/safety 분야에서 많이 사용하는 방법:

```text
FMEA
=
Failure Modes and Effects Analysis
```

이다.

---

# 93. FMEA의 핵심 질문

각 component에 대해:

```text
어떻게 실패할 수 있는가?
그 failure의 영향은?
어떻게 감지할 것인가?
어떻게 대응할 것인가?
```

를 정리한다.

---

# 94. FMEA Example

| Component | Failure Mode | Effect | Detection | Response |
|---|---|---|---|---|
| LiDAR | Data loss | SLAM degraded | Topic 0 Hz | Stop autonomy |
| Camera | Frame loss | Vision unavailable | FPS 0 | LiDAR-only mode |
| SSD | Disk full | Logging fails | Disk >95% | Stop optional logs |
| FAST-LIO2 | Process crash | No localization | Heartbeat lost | Restart once |

---

# 95. Severity

Failure가 얼마나 심각한 영향을 주는지:

```text
Severity
```

를 평가할 수 있다.

---

# 96. Occurrence

Failure가 얼마나 자주 발생할 가능성이 있는지:

```text
Occurrence
```

를 평가할 수 있다.

---

# 97. Detectability

Failure를 얼마나 쉽게 감지할 수 있는지:

```text
Detectability
```

를 평가할 수 있다.

---

# 98. Fault Tree

Failure의 원인을 tree 형태로 분석하는 방법:

```text
Fault Tree Analysis
```

도 있다.

---

# 99. Example Fault Tree

```text
Localization Lost
       │
       ├── LiDAR Failure
       │
       ├── IMU Failure
       │
       ├── Time Sync Failure
       │
       ├── FAST-LIO2 Crash
       │
       └── CPU Overload
```

---

# 100. Root Cause와 Symptom

예:

```text
Navigation stopped
```

는 symptom일 수 있다.

Root cause:

```text
Ethernet cable intermittent
```

일 수 있다.

Observability와 fault analysis를 함께 봐야 한다.

---

# 101. MTBF

MTBF:

```text
Mean Time Between Failures
```

이다.

Repair 가능한 system에서 failure 사이의 평균 시간을 나타내는 데 사용된다.

---

# 102. MTTF

MTTF:

```text
Mean Time To Failure
```

이다.

Repair하지 않는 component의 평균 failure 시간 등을 표현할 때 사용한다.

---

# 103. MTTR

MTTR:

```text
Mean Time To Repair
```

또는 운영 문맥에 따라 recovery 의미로 쓰이기도 한다.

여기서는:

```text
Failure 발생 후 정상 복구까지 걸리는 평균 시간
```

으로 이해하면 된다.

---

# 104. Availability

Availability는 단순화하면:

```text
얼마나 자주 system을 사용할 수 있는가?
```

다.

개념적으로:

```text
Availability
≈
Uptime / Total Time
```

---

# 105. Availability와 MTBF/MTTR

매우 단순화하면:

```text
Availability
≈
MTBF / (MTBF + MTTR)
```

관계를 생각할 수 있다.

즉:

```text
고장을 줄이는 것
+
복구 시간을 줄이는 것
```

둘 다 availability를 높인다.

---

# 106. Recovery Time Objective

RTO:

```text
Recovery Time Objective
```

같은 개념을 운영에서 사용할 수 있다.

예:

```text
FAST-LIO2 failure 후
5초 이내 recovery
```

같은 목표를 정의할 수 있다.

---

# 107. Persistent State

Restart 후에도 유지해야 하는 데이터:

```text
Map
Config
Calibration
Mission state
```

가 있을 수 있다.

Recovery를 설계할 때 어떤 state를 보존해야 하는지 생각해야 한다.

---

# 108. Stateless Component

Restart해도 잃을 중요한 내부 상태가 적은 component는
복구하기 쉽다.

```text
Stateless
```

하게 설계하면 reliability에 도움이 될 수 있다.

---

# 109. Stateful Component

SLAM처럼 내부에:

```text
Map
Filter state
Trajectory
```

를 가진 component는 restart가 더 복잡하다.

---

# 110. Checkpoint

중간 state를 저장해 두는 것을:

```text
Checkpoint
```

라고 한다.

Failure 후 처음부터 시작하지 않고
최근 checkpoint에서 복구할 수 있다.

---

# 111. Checkpoint Trade-off

자주 저장하면:

```text
Recovery loss ↓
```

하지만:

```text
Storage I/O ↑
Performance overhead ↑
```

가 생긴다.

---

# 112. State Persistence Example

Mapping 중:

```text
Map snapshot every 5 min
```

을 저장하면 crash 후 일부 작업을 복구할 수 있다.

---

# 113. Transaction

여러 변경을 하나의 단위로 처리해:

```text
전부 성공
or
전부 실패
```

하도록 하는 개념이:

```text
Transaction
```

이다.

Config/update consistency에서 유용하다.

---

# 114. Atomic Configuration Update

Config file을 직접 덮어쓰다가 power loss가 나면
file이 깨질 수 있다.

더 안전한 방식:

```text
Write new file
     ↓
Verify
     ↓
Atomic rename/switch
```

같은 패턴을 사용할 수 있다.

---

# 115. Safe Update와 Reliability

Chapter 17의 OTA도 reliability 문제다.

```text
Download
Verify
Install
Health Check
Rollback
```

이 모두 fault-tolerant deployment를 위한 구조다.

---

# 116. Network Partition

Robot과 cloud가 서로 통신할 수 없는 상황:

```text
Network Partition
```

이 발생할 수 있다.

Robot은 이런 상황을 정상적인 failure mode로 취급해야 한다.

---

# 117. Cloud가 없어도 동작

좋은 edge design:

```text
Cloud unavailable
        ↓
Robot still:
SLAM
Navigation
Safety
```

핵심 기능을 계속 수행할 수 있다.

---

# 118. Local Autonomy

Edge computing의 중요한 장점이다.

```text
Cloud
   X

Robot
├── Perception
├── SLAM
├── Navigation
└── Safety
```

local에서 계속 동작한다.

---

# 119. Network Reconnection

Cloud가 다시 연결되면:

```text
Buffered telemetry
Logs
Events
```

를 다시 upload할 수 있다.

---

# 120. Duplicate Message

Reconnect 과정에서 같은 telemetry/event가 두 번 전달될 수 있다.

그래서:

```text
Message ID
Sequence Number
Idempotency
```

가 중요할 수 있다.

---

# 121. Sensor Failure Modes

Sensor는 단순히:

```text
ON
OFF
```

만 있는 것이 아니다.

Failure mode:

```text
No data
Wrong data
Frozen data
Delayed data
Noisy data
Intermittent data
```

가 있을 수 있다.

---

# 122. Frozen Sensor

예:

```text
IMU value:
0.01
0.01
0.01
0.01
...
```

계속 message는 오지만 값이 멈춰 있을 수 있다.

따라서:

```text
Topic Hz > 0
```

만으로 health를 판단하면 부족하다.

---

# 123. Stale Data

Message가 오지만 timestamp가 오래된 경우:

```text
Stale Data
```

다.

Chapter 18의 message age monitoring과 연결된다.

---

# 124. Plausible but Wrong

가장 어려운 failure:

```text
Sensor data가 그럴듯하지만 틀림
```

이다.

예:

```text
IMU bias drift
LiDAR calibration shift
```

health monitoring이 더 어렵다.

---

# 125. Calibration Failure

Sensor mounting이 움직이면:

```text
Extrinsic changed
```

할 수 있다.

Sensor 자체는 정상인데 SLAM 결과가 나빠진다.

이것도 failure mode다.

---

# 126. Time Sync Failure

NTP/PTP 문제가 생기면:

```text
LiDAR data normal
IMU data normal
```

이어도 fusion 결과가 틀릴 수 있다.

즉 Chapter 11의 time sync도 reliability 대상이다.

---

# 127. Silent Failure

명확한 error 없이 결과만 틀리는 failure:

```text
Silent Failure
```

가 가장 위험한 경우 중 하나다.

---

# 128. Fail-Stop Failure

Component가 문제 발생 시 완전히 멈추는 형태:

```text
Fail-Stop
```

는 오히려 감지하기 쉬운 편이다.

---

# 129. Byzantine Failure

Distributed system에서는 component가
예측 불가능하거나 일관되지 않은 잘못된 값을 보내는 failure를:

```text
Byzantine Failure
```

라고 부른다.

일반 robot system에서 자주 직접 구현하는 개념은 아니지만
failure가 항상 단순 crash는 아니라는 점을 이해하는 데 유용하다.

---

# 130. Safety Monitor

Main autonomy와 별도로
system 상태를 감시하는 safety monitor를 둘 수 있다.

```text
Autonomy
   │
   ▼
Command
   │
   ▼
Safety Monitor
   │
   ▼
Controller
```

---

# 131. Independent Safety Path

가능하면 safety mechanism이 main application failure에 같이 죽지 않도록 설계한다.

예:

```text
Jetson crash
     ↓

MCU still detects command timeout
     ↓
Safe stop
```

---

# 132. Command Timeout

Jetson에서 velocity command가 더 이상 오지 않는다면:

```text
Last command age > threshold
```

일 때 MCU가 robot을 stop할 수 있다.

---

# 133. Stale Command 위험

마지막 command가:

```text
Walk forward
```

였는데 Jetson이 죽으면
이를 계속 실행하면 위험하다.

따라서 command freshness가 중요하다.

---

# 134. E-stop

Emergency Stop은 reliability/safety architecture의 마지막 보호 수단 중 하나다.

개념적으로:

```text
Operator
   ↓
E-stop
   ↓
Safety system
   ↓
Actuation stop
```

Main autonomy software와 독립적인 path가 중요한 경우가 많다.

---

# 135. Fail-Silent

Component가 이상할 때 잘못된 값을 계속 보내는 대신
output을 중단하는 동작을:

```text
Fail-Silent
```

라고 한다.

잘못된 command를 계속 내는 것보다 안전할 수 있다.

---

# 136. Safety vs Reliability

둘은 같은 개념이 아니다.

```text
Reliability
→ 정상 기능을 얼마나 잘 유지하는가?

Safety
→ 위험한 상태를 얼마나 잘 피하는가?
```

이다.

신뢰성이 낮아도 안전하게 멈출 수 있고,
신뢰성이 높아도 잘못된 동작이 위험할 수 있다.

---

# 137. Availability vs Safety

예:

```text
Localization uncertain
```

인데 계속 움직이면 availability는 높아 보일 수 있지만
safety가 낮아질 수 있다.

따라서 critical robot에서는 safety가 우선일 수 있다.

---

# 138. Reliability Requirement

Component마다 요구 수준을 다르게 정의할 수 있다.

예:

```text
Cloud telemetry
→ temporary failure acceptable

Camera AI
→ degraded mode possible

Localization
→ critical

Low-level safety controller
→ highly critical
```

---

# 139. Failure Budget

System이 어느 정도 failure를 허용할 수 있는지 정할 수 있다.

예:

```text
Telemetry:
1% packet loss acceptable

Control:
deadline miss almost unacceptable
```

---

# 140. Error Budget

SRE에서 사용하는:

```text
Error Budget
```

개념도 reliability 목표와 연결할 수 있다.

예:

```text
SLO = 99.9%
```

라면 남은 0.1%의 failure allowance를 error budget으로 볼 수 있다.

---

# 141. Reliability Testing

정상 테스트 외에:

```text
Long-duration test
Reboot test
Network loss test
Sensor disconnect test
Disk full test
CPU overload test
```

를 해야 한다.

---

# 142. Soak Test

System을 오랫동안 계속 실행하는 테스트:

```text
Soak Test
```

이다.

예:

```text
24 hours
72 hours
1 week
```

Memory leak, thermal issue, log growth를 찾는 데 유용하다.

---

# 143. Stress Test

System에 높은 workload를 주는 테스트:

```text
Stress Test
```

예:

```text
Maximum camera
Large point cloud
rosbag recording
AI inference
```

를 동시에 실행한다.

---

# 144. Endurance Test

장시간 실제 operating condition에서
system이 안정적으로 동작하는지 확인한다.

Robot 현장 deployment 전에 중요하다.

---

# 145. Reboot Test

Robot을 여러 번 reboot하면서:

```text
Network comes up?
Docker starts?
ROS nodes start?
Sensors reconnect?
```

를 확인한다.

---

# 146. Power Cycle Test

전원을 완전히 껐다 켜는 test도 중요하다.

단순 software reboot와 다른 hardware initialization 문제를 찾을 수 있다.

---

# 147. Dependency Failure Test

예:

```text
Start FAST-LIO2 before LiDAR
```

했을 때 system이 어떻게 행동하는지 본다.

잘 설계하면:

```text
Wait
Retry
Become Ready
```

할 수 있다.

---

# 148. Startup Dependency

```text
Network
   ↓
Sensor
   ↓
Driver
   ↓
SLAM
   ↓
Navigation
```

dependency를 명확하게 관리해야 한다.

---

# 149. Readiness

Component가 process로 시작됐다고 바로 준비된 것은 아니다.

예:

```text
FAST-LIO2 started
but waiting for IMU
```

이면:

```text
Alive
but not Ready
```

다.

Chapter 18과 연결된다.

---

# 150. Reliability Architecture

예:

```text
                         Robot

                ┌──────────────────┐
                │ Health Monitor   │
                └────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      LiDAR           FAST-LIO2        Camera
         │               │               │
         │ health        │ heartbeat     │ FPS
         └───────────────┼───────────────┘
                         ▼
                   Fault Manager
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
             Retry     Degrade     Safe Stop
```

---

# 151. Fault Manager

여러 failure를 중앙에서 판단하는 component를:

```text
Fault Manager
```

형태로 설계할 수 있다.

역할:

```text
Collect health
Classify fault
Choose recovery
Escalate if needed
```

---

# 152. Local Recovery vs Central Recovery

각 component가 스스로 recovery할 수도 있다.

```text
Camera node
→ reconnect
```

또는 중앙 supervisor가:

```text
Restart camera node
```

할 수도 있다.

둘의 책임을 명확히 해야 한다.

---

# 153. Recovery Escalation

예:

```text
Camera timeout
   ↓
Reconnect
   ↓ fail
Restart driver
   ↓ fail
Restart container
   ↓ fail
Mark degraded
```

처럼 단계적으로 올라간다.

---

# 154. Recovery가 성공했는지 확인

Restart했다고 끝이 아니다.

```text
Restart
  ↓
Health Check
  ↓
Healthy?
```

를 확인해야 한다.

---

# 155. Recovery Loop

전체:

```text
Detect
   ↓
Diagnose
   ↓
Recover
   ↓
Verify
   ↓
Resume
```

이다.

---

# 156. Incident Logging

Failure가 발생하면 반드시 기록한다.

예:

```text
Timestamp
Robot ID
Fault
Detection method
Recovery action
Recovery result
Software version
```

---

# 157. Fleet Reliability

Robot 한 대뿐 아니라 fleet 전체도 본다.

예:

```text
100 Robots

97 Healthy
2 Degraded
1 Offline
```

---

# 158. Correlated Failure

여러 robot이 동시에 실패하면
개별 robot 문제보다 공통 원인을 의심해야 한다.

예:

```text
New software deployment
Cloud outage
Site network outage
Extreme temperature
```

---

# 159. Fleet-Wide Rollback

새 version 배포 후:

```text
30% robots fail
```

한다면 rollout을 중지하고 rollback할 수 있어야 한다.

Chapter 17과 연결된다.

---

# 160. Deployment Failure도 Fault다

Reliability는 runtime hardware failure만 다루는 것이 아니다.

```text
Bad Config
Bad Software
Bad Update
Expired Certificate
```

도 failure source다.

---

# 161. Human Error

현실에서 중요한 failure source:

```text
Human Error
```

이다.

예:

```text
Wrong config
Wrong robot updated
Cable connected incorrectly
Wrong command
```

---

# 162. Human Error 줄이기

```text
Automation
Validation
Confirmation
Version Control
Access Control
Checklists
```

로 줄일 수 있다.

---

# 163. Configuration Validation

잘못된 config를 실행 전에 검사한다.

예:

```text
LiDAR IP valid?
Map exists?
ROS_DOMAIN_ID valid?
Calibration file exists?
```

---

# 164. Preflight Check

Mission 시작 전:

```text
Sensors healthy?
Localization valid?
Battery enough?
Disk enough?
Temperature normal?
Network required?
```

를 확인한다.

---

# 165. Go / No-Go

Preflight 결과에 따라:

```text
GO
```

또는:

```text
NO-GO
```

를 결정할 수 있다.

---

# 166. Fail Early

문제가 명확하다면 시작 후 이상 동작하는 것보다
처음부터 실행을 막는 것이 좋다.

예:

```text
Calibration missing
      ↓
Do not start autonomy
```

---

# 167. Defensive Programming

잘못된 input이나 예상치 못한 상황을 고려해 code를 작성한다.

예:

```text
null check
range check
timeout
exception handling
```

---

# 168. Assertion

개발 단계에서는:

```text
이 조건은 반드시 참이어야 한다.
```

를 assertion으로 확인할 수 있다.

Production에서는 assertion failure를 어떻게 처리할지도 생각해야 한다.

---

# 169. Exception Handling

예외를 무조건 무시하면 안 된다.

나쁜 방식:

```text
catch (...) {
}
```

좋은 방식:

```text
Log
Classify
Recover or Fail Safely
```

---

# 170. Fail Fast

복구할 수 없는 잘못된 상태에서
계속 잘못 동작하는 것보다 빠르게 failure를 명확히 드러내는 전략:

```text
Fail Fast
```

이 유용할 수 있다.

---

# 171. Silent Data Corruption

가장 위험한 문제 중 하나:

```text
System appears alive
but data is wrong
```

이다.

예:

```text
Corrupted map
Bad calibration
Wrong timestamp
```

---

# 172. Checksum

Stored/transferred data corruption을 확인하기 위해 checksum/hash를 사용할 수 있다.

예:

```bash
sha256sum
```

Chapter 13, 16과 연결된다.

---

# 173. ECC Memory

일부 system에서는:

```text
ECC
=
Error-Correcting Code
```

memory를 사용해 memory bit error를 감지/보정할 수 있다.

Hardware reliability와 관련된다.

---

# 174. Bit Flip

Radiation, electrical noise 등으로 memory bit가 의도치 않게 바뀔 수 있다.

```text
0 → 1
```

이런 현상을 bit flip이라고 한다.

---

# 175. Industrial Environment

건설/조선 현장에서는:

```text
Dust
Vibration
Heat
Cold
Water
EMI
Power fluctuation
```

등이 reliability에 영향을 준다.

Software만 잘 짠다고 충분하지 않다.

---

# 176. Environmental Qualification

Hardware를 실제 operating environment에서 검증해야 한다.

예:

```text
Temperature
Shock
Vibration
Ingress
EMC
```

등.

---

# 177. Connector Reliability

Robot에서는 cable/connector가 흔한 failure source다.

```text
Vibration
Repeated movement
Dust
Loose connector
```

로 intermittent failure가 발생할 수 있다.

---

# 178. Intermittent Fault

계속 고장난 것이 아니라 가끔 발생하는 문제:

```text
Intermittent Fault
```

이다.

가장 debugging하기 어려운 유형 중 하나다.

---

# 179. Example

```text
LiDAR
works 10 min
drops 1 sec
works again
```

일 수 있다.

그래서 long-term logs와 counters가 중요하다.

---

# 180. Fault Counter

예:

```text
LiDAR reconnect count
Network reset count
FAST-LIO2 restart count
```

를 metric으로 관리하면 intermittent problem을 찾는 데 도움이 된다.

---

# 181. Recovery Counter

Component가 하루에:

```text
Restart 30 times
```

인데 현재는 healthy라면
단순 현재 상태만 보고 정상이라고 하면 안 된다.

---

# 182. Reliability Trend

시간에 따라:

```text
Failures/day
Restarts/day
Packet errors
Temperature
```

trend를 본다.

---

# 183. Predictive Maintenance

이런 trend를 이용해 failure 전에 maintenance할 수 있다.

예:

```text
SSD error increasing
Fan speed abnormal
Battery degradation
Connector error count increasing
```

---

# 184. Maintenance

Reliability는 software recovery만이 아니다.

```text
Cleaning
Cable inspection
Fan replacement
SSD replacement
Calibration
```

같은 physical maintenance도 포함된다.

---

# 185. Preventive Maintenance

고장 전에 주기적으로 maintenance:

```text
Preventive Maintenance
```

---

# 186. Corrective Maintenance

고장난 후 수리:

```text
Corrective Maintenance
```

---

# 187. Predictive Maintenance

데이터를 기반으로 고장 가능성을 예측해서 maintenance:

```text
Predictive Maintenance
```

---

# 188. Reliability Dashboard

예:

```text
Vision60-001

Status:
DEGRADED

LiDAR:
Healthy

Camera:
Failed

SLAM:
Healthy

Restart Count:
2

Disk:
72%

Temperature:
68°C

Last Failure:
Camera timeout 13:42
```

---

# 189. Reliability KPI

예:

```text
Mission Success Rate
Localization Uptime
Failure Count / Hour
Automatic Recovery Rate
MTTR
Unexpected Reboot Count
```

등.

---

# 190. Automatic Recovery Rate

예:

```text
100 failures

85 automatically recovered
15 required human intervention
```

이면 automatic recovery rate:

```text
85%
```

라고 볼 수 있다.

---

# 191. Human Intervention Rate

운영 효율을 위해:

```text
몇 번이나 사람이 직접 robot에 가야 했는가?
```

도 중요한 metric일 수 있다.

---

# 192. Remote Recoverability

현장에 직접 가지 않고:

```text
Remote restart
Remote update
Remote diagnostics
```

로 복구할 수 있는 능력도 fleet reliability에 중요하다.

---

# 193. But Remote Access Can Fail

Network 자체가 failure라면 remote recovery가 불가능하다.

그래서 local autonomous recovery가 중요하다.

---

# 194. Local vs Remote Recovery

```text
Local
→ Watchdog
→ Automatic restart
→ Safe stop

Remote
→ Operator diagnosis
→ Config fix
→ Software update
```

둘 다 필요하다.

---

# 195. Recovery Priority

예:

```text
1. Safety
2. Stabilize system
3. Restore critical function
4. Restore optional function
5. Upload diagnostics
```

순서로 생각할 수 있다.

---

# 196. Reliability Design 질문

각 component마다 다음을 묻는다.

```text
How can it fail?

How do we detect it?

How fast can we detect it?

What is the impact?

Can we continue without it?

Can we recover automatically?

What is the safe state?

What logs do we need?
```

---

# 197. Vision60 Example

개념적인 fault handling:

```text
                    Vision60

LiDAR ───────┐
IMU ─────────┤
Camera ──────┤
Jetson ──────┤
Network ─────┤
             ▼
       Health Monitor
             │
             ▼
        Fault Manager
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   Retry   Degrade   Stop
     │       │        │
     ▼       ▼        ▼
 Recover   Limited   Safe
          Operation  State
```

---

# 198. FAST-LIO2 Failure Example

```text
FAST-LIO2 /odometry stops
      ↓
Health Monitor detects 0 Hz
      ↓
Check LiDAR / IMU
      │
      ├── Sensors healthy
      │       ↓
      │   Restart FAST-LIO2
      │
      └── LiDAR failed
              ↓
         Stop autonomy
```

---

# 199. Network Failure Example

```text
Company Wi-Fi Lost
      ↓
Cloud Offline
      ↓
Fleet telemetry stopped
      ↓
Local autonomy continues
      ↓
Store logs locally
      ↓
Reconnect later
```

---

# 200. Disk Failure Example

```text
Disk > 90%
     ↓
Stop optional camera recording
     ↓
Preserve critical logs
     ↓
Alert operator
```

---

# 201. Thermal Failure Example

```text
Temperature rising
      ↓
Warning
      ↓
Reduce optional workload
      ↓
Temperature still high
      ↓
Safe shutdown / degraded mode
```

정확한 threshold는 hardware specification을 사용해야 한다.

---

# 202. Power Failure Example

```text
Battery low
    ↓
Stop non-critical workload
    ↓
Return / Safe position
    ↓
Shutdown cleanly
```

---

# 203. Reliability Layer

전체 stack의 각 layer에서 failure가 생길 수 있다.

```text
Hardware
   ↓
Driver
   ↓
Linux
   ↓
Network
   ↓
Docker
   ↓
ROS 2
   ↓
SLAM / AI
   ↓
Control
```

따라서 reliability는 특정 하나의 layer만의 문제가 아니다.

---

# 204. Chapter 10과 연결

Chapter 10에서는:

```text
문제가 생기면 어떻게 debugging할까?
```

를 배웠다.

Chapter 19에서는:

```text
문제가 생기기 전에 어떻게 감지하고,
생겼을 때 system이 스스로 어떻게 대응할까?
```

를 배운다.

---

# 205. Chapter 18과 연결

Chapter 18:

```text
Observe
Detect
Alert
```

Chapter 19:

```text
Respond
Recover
Degrade
Fail Safely
```

이다.

---

# 206. Chapter 17과 연결

Chapter 17:

```text
Fleet Deployment
Rollback
Remote Management
```

Chapter 19:

```text
Failure Recovery
Fault Tolerance
Reliability
```

가 결합된다.

---

# 207. Mini Practice 1

Vision60의 가능한 failure mode 10개를 적어본다.

예:

```text
LiDAR disconnect
IMU failure
Camera failure
Xavier crash
Orin crash
Disk full
Overtemperature
Network loss
FAST-LIO2 crash
Time synchronization failure
```

---

# 208. Mini Practice 2

각 failure에 대해 다음을 적는다.

```text
Detection
Impact
Recovery
Safe State
```

---

# 209. Mini Practice 3

예:

```text
Failure:
LiDAR 0 Hz
```

다음 질문:

```text
LiDAR hardware failure인가?

Network failure인가?

Driver failure인가?

ROS failure인가?
```

어떻게 구분할지 생각한다.

---

# 210. Mini Practice 4

다음 state machine을 직접 작성한다.

```text
NORMAL
 ↓
DEGRADED
 ↓
SAFE_STOP
```

어떤 조건에서 state가 변경되는지 정의한다.

---

# 211. Mini Practice 5

Fault injection test:

```text
Kill FAST-LIO2 process
```

를 가정한다.

원하는 system behavior:

```text
Detect within ? seconds
Restart how many times?
When to give up?
When to safe stop?
What log to save?
```

를 정한다.

---

# 212. Mini Practice 6

현재 architecture에서 SPOF를 찾는다.

예:

```text
Single Jetson?
Single LiDAR?
Single network switch?
Single power rail?
```

---

# 213. Mini Practice 7

다음 중 graceful degradation 가능한 것을 구분한다.

```text
Camera lost
Cloud lost
LiDAR lost
Logging lost
Battery critical
```

각 상황에서 어떤 기능을 유지할 수 있는지 적는다.

---

# 214. Mini Practice 8

FMEA table을 직접 만든다.

| Component | Failure | Detection | Impact | Response |
|---|---|---|---|---|
| LiDAR | No packets | 0 Hz | Localization risk | Stop autonomy |
| Camera | No frames | 0 FPS | No vision AI | Degraded |
| Cloud | Offline | Heartbeat | No fleet link | Continue local |
| SSD | Full | >95% | Logging loss | Drop optional logs |

---

# 215. 반드시 구분할 것

```text
Fault
≠
Error
≠
Failure

Reliability
≠
Fault Tolerance

Redundancy
≠
Backup only

Fail-Safe
≠
Fail-Operational

Retry
≠
Recovery

Watchdog
≠
Health Check

Alive
≠
Correct

No Error Log
≠
Healthy

Availability
≠
Safety

Redundancy
≠
No Common-Mode Failure
```

---

# 216. Reliability Mental Model

```text
Fault
  │
  ▼
Detect
  │
  ▼
Isolate
  │
  ▼
Assess Severity
  │
  ├── Retry
  ├── Restart
  ├── Switch Backup
  ├── Degrade
  └── Safe Stop
  │
  ▼
Verify Recovery
  │
  ▼
Return to Service
```

---

# 217. Robot Reliability Mental Model

```text
                   Robot System

                         │
                         ▼
                   Normal Operation
                         │
                         ▼
                       Fault
                         │
                         ▼
                     Detection
                         │
                         ▼
                 Fault Management
                         │
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
        Recover        Degrade         Safe Stop
          │              │               │
          └──────────────┼───────────────┘
                         ▼
                    Verify State
                         │
                         ▼
                    Continue / Alert
```

---

# 218. Chapter 19 핵심

Reliable robot은:

```text
절대 고장나지 않는 robot
```

이 아니다.

현실적으로는:

```text
문제를 빨리 감지하고,
영향을 제한하고,
가능하면 자동 복구하고,
복구할 수 없으면 안전하게 동작하는 robot
```

이다.

즉:

```text
Detect
+
Isolate
+
Recover
+
Degrade
+
Fail Safely
```

가 핵심이다.

---

# Edge Computing Course Complete

## Chapter 1 ~ Chapter 19

```text
Chapter 1
Computer Hardware Basics

Chapter 2
ARM vs x86

Chapter 3
Linux for Edge Computers

Chapter 4
NVIDIA Jetson & JetPack

Chapter 5
Hardware Interfaces

Chapter 6
ROS 2 as a Robotics Middleware

Chapter 7
CUDA & TensorRT for Robotics

Chapter 8
Networking for Robots

Chapter 9
Docker on Jetson

Chapter 10
ROS 2 + Jetson Debugging & Deployment

Chapter 11
Time Synchronization & Sensor Timing

Chapter 12
Power, Thermal & Performance Management

Chapter 13
Storage & Data Logging

Chapter 14
Device Drivers & Kernel Basics

Chapter 15
Real-Time Computing

Chapter 16
Embedded Security

Chapter 17
Remote Deployment & Fleet Management

Chapter 18
Observability & Monitoring

Chapter 19
Reliability & Fault Tolerance
```

---

# 전체 Edge Computing Mental Model

```text
                         Robot

                    Sensors / Motors
                           │
                           ▼
                   Hardware Interface
               Ethernet / CAN / USB
                           │
                           ▼
                     Linux Kernel
                           │
                           ▼
                   Drivers / Runtime
                           │
                           ▼
                       ROS 2
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
           SLAM         Perception     Navigation
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                         Robot

그리고 전체 system을 둘러싸는 것:

Power / Thermal
Storage
Timing
Security
Deployment
Monitoring
Reliability
```

---

# 다음 Study Track

Edge Computing 트랙은 여기서 마무리하고,
다음은 별도 `_study/slam/` 트랙으로 넘어가는 것이 좋다.

추천 순서:

```text
SLAM Chapter 1
Coordinate Frames & TF

SLAM Chapter 2
Rotation Matrix / Euler / Quaternion

SLAM Chapter 3
Rigid Body Transformations

SLAM Chapter 4
IMU Fundamentals

SLAM Chapter 5
State Estimation

SLAM Chapter 6
Kalman Filter

SLAM Chapter 7
EKF / ESKF

SLAM Chapter 8
LiDAR Odometry

SLAM Chapter 9
FAST-LIO2

SLAM Chapter 10
Mapping / Local Map / Global Map

SLAM Chapter 11
Loop Closure / Pose Graph Optimization

SLAM Chapter 12
Leg Odometry

SLAM Chapter 13
Contact / Gait-Aware Estimation

SLAM Chapter 14
Quadruped SLAM Architecture

SLAM Chapter 15
Vision60 FAST-LIO2 Code Walkthrough
```

이제 Edge Computing 쪽은 **“Jetson 기반 로봇이 실제 현장에서 안정적으로 돌아가게 하는 컴퓨팅 시스템 전체”**를 한 번 훑은 상태라고 보면 돼.