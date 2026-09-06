---
title: "Chapter 18. Observability & Monitoring"
importance: 19
---

> **Goal:** Robot이 단순히 "켜져 있는지"가 아니라,
> 내부에서 실제로 어떤 일이 일어나고 있는지 외부에서 파악할 수 있도록 만드는 방법을 이해한다.
>
> Logs, Metrics, Traces, Events, Health Check, Dashboard, Alerting의 차이를 이해하고,
> CPU/GPU/RAM, Temperature, Disk, Network, ROS Topic Rate, Latency를
> 실제 robot monitoring에 연결하는 것이 목표다.

---

# 1. Monitoring이 왜 필요한가?

Robot 한 대를 직접 보고 있다면:

```text
SSH 접속
htop 실행
ros2 topic list
tegrastats 확인
```

하면서 상태를 볼 수 있다.

하지만 robot이 여러 대라면:

```text
Robot 001
Robot 002
Robot 003
...
Robot 100
```

모두 직접 SSH로 접속해서 확인하기 어렵다.

그래서 robot이 자신의 상태를 외부로 알려줘야 한다.

---

# 2. Observability란?

Observability는:

> System 외부에서 수집한 정보만으로 내부 상태를 얼마나 잘 이해할 수 있는가?

를 의미한다.

예:

```text
Robot stopped
```

만 알면 원인을 알기 어렵다.

하지만:

```text
CPU 95%
LiDAR 0 Hz
Disk 97%
FAST-LIO2 error
Temperature 88°C
```

를 함께 볼 수 있다면 원인을 훨씬 빨리 좁힐 수 있다.

---

# 3. Monitoring과 Observability는 같은가?

비슷하지만 완전히 같지는 않다.

```text
Monitoring
→ 미리 정의한 상태/metric을 계속 관찰

Observability
→ 예상하지 못한 문제도 내부 정보를 조합해 원인을 추론할 수 있는 능력
```

이라고 이해하면 된다.

---

# 4. Observability의 대표적인 세 요소

소프트웨어 시스템에서는 흔히:

```text
Logs
Metrics
Traces
```

를 observability의 핵심 요소로 본다.

Robot에서는 여기에:

```text
Events
Health
Sensor status
```

도 매우 중요하다.

---

# 5. Logs

Logs는:

> 특정 시점에 어떤 일이 일어났는지 기록한 text/event data

다.

예:

```text
23:01:10 FAST-LIO2 started
23:01:15 LiDAR timeout
23:01:16 Mapping paused
```

---

# 6. Logs의 장점

Logs는 상세한 context를 남길 수 있다.

예:

```text
error code
file name
sensor state
exception
configuration value
```

문제 원인 분석에 매우 유용하다.

---

# 7. Logs의 단점

Log를 너무 많이 남기면:

```text
Disk 사용량 증가
I/O 증가
분석 어려움
```

문제가 생긴다.

그래서 log level과 retention을 적절히 관리해야 한다.

---

# 8. Metrics

Metrics는:

> 숫자로 표현되는 system 상태 값

이다.

예:

```text
CPU Usage = 65%
GPU Usage = 80%
RAM Usage = 12 GB
Temperature = 72°C
LiDAR Rate = 10 Hz
Odometry Rate = 95 Hz
```

---

# 9. Metrics의 특징

Metrics는 시간에 따라 계속 기록할 수 있다.

예:

```text
Time    CPU

10:00   30%
10:01   45%
10:02   90%
10:03   95%
```

그래서 trend를 보기 좋다.

---

# 10. Time Series

시간에 따라 변하는 metric data를:

```text
Time Series
```

라고 한다.

예:

```text
Temperature over time
CPU usage over time
Latency over time
```

---

# 11. Traces

Trace는 하나의 요청이나 작업이
여러 component를 거치는 경로를 추적하는 것이다.

예:

```text
Camera Frame
   ↓
Driver
   ↓
ROS 2
   ↓
Preprocessing
   ↓
Inference
   ↓
Postprocessing
   ↓
Navigation
```

각 단계의 시간을 기록하면 어디서 느려졌는지 알 수 있다.

---

# 12. Distributed Trace

여러 process나 computer에 걸쳐 하나의 작업을 추적하면:

```text
Distributed Trace
```

라고 한다.

예:

```text
Xavier
LiDAR Driver
   ↓
ROS 2 Network
   ↓
Orin
FAST-LIO2
   ↓
Navigation
```

---

# 13. Events

Event는 중요한 사건이 발생했을 때 기록하는 데이터다.

예:

```text
E-stop pressed
LiDAR disconnected
Localization lost
Deployment started
Robot rebooted
```

---

# 14. Event vs Metric

Metric:

```text
Temperature = 75°C
```

Event:

```text
Temperature exceeded threshold
```

이다.

---

# 15. Event vs Log

Event는 중요한 상태 변화 중심이고,
log는 더 상세한 실행 정보를 담을 수 있다.

예:

```text
Event
Localization Lost

Log
ICP residual exceeded threshold...
```

---

# 16. Health Check

Health Check는:

> Component가 정상적으로 동작하고 있는지 판단하는 검사

이다.

예:

```text
LiDAR healthy?
FAST-LIO2 healthy?
Navigation healthy?
Disk healthy?
```

---

# 17. Process Alive만 보면 부족하다

예:

```text
FAST-LIO2 process exists
```

하지만:

```text
/odometry = 0 Hz
```

일 수 있다.

따라서:

```text
Process alive
≠
Application healthy
```

이다.

---

# 18. Liveness

Liveness는:

> Component가 살아 있는가?

를 본다.

예:

```text
Process running
Heartbeat received
```

---

# 19. Readiness

Readiness는:

> Component가 실제 작업을 수행할 준비가 되어 있는가?

를 본다.

예:

```text
FAST-LIO2 process alive
but LiDAR not connected
```

이면:

```text
Alive
but Not Ready
```

일 수 있다.

---

# 20. Liveness vs Readiness

```text
Liveness
→ 살아 있는가?

Readiness
→ 실제 일을 할 준비가 되었는가?
```

이다.

---

# 21. Robot Health는 계층적으로 봐야 한다

예:

```text
Robot Health

├── System
│   ├── CPU
│   ├── RAM
│   └── Disk
│
├── Hardware
│   ├── LiDAR
│   ├── IMU
│   └── Camera
│
├── ROS
│   ├── Nodes
│   ├── Topics
│   └── DDS
│
└── Application
    ├── SLAM
    ├── Navigation
    └── AI
```

---

# 22. System Metrics

먼저 computer 자체 상태를 본다.

```text
CPU Usage
GPU Usage
RAM Usage
Disk Usage
Temperature
Power
```

---

# 23. CPU Monitoring

Linux:

```bash
top
```

또는:

```bash
htop
```

을 사용할 수 있다.

Fleet monitoring에서는 이 값을 telemetry로 보내면 된다.

---

# 24. GPU Monitoring

Jetson에서는:

```bash
tegrastats
```

를 사용할 수 있다.

예:

```text
GPU utilization
Memory
Temperature
Power
```

등을 볼 수 있다.

---

# 25. Temperature Monitoring

온도는 매우 중요하다.

예:

```text
CPU 85°C
GPU 88°C
```

가 계속된다면 thermal throttling 가능성을 봐야 한다.

---

# 26. Disk Monitoring

확인:

```bash
df -h
```

Fleet metric:

```text
Disk Used %
Disk Available GB
```

를 보낼 수 있다.

---

# 27. Disk Alert

예:

```text
Warning:
Disk > 80%

Critical:
Disk > 95%
```

처럼 threshold를 둘 수 있다.

정확한 threshold는 application에 맞춰 결정한다.

---

# 28. RAM Monitoring

```bash
free -h
```

로 확인할 수 있다.

중요:

```text
Used RAM
Available RAM
Swap
```

을 함께 본다.

---

# 29. OOM 위험

RAM 사용량이 계속 증가하면:

```text
Memory Leak
Queue buildup
Large cache
```

를 의심할 수 있다.

결국 OOM killer가 process를 종료할 수 있다.

---

# 30. Network Monitoring

Network에서는:

```text
Bandwidth
Packet Loss
Errors
Dropped Packets
Latency
```

를 볼 수 있다.

---

# 31. Interface Statistics

Linux:

```bash
ip -s link
```

예:

```text
RX packets
TX packets
errors
dropped
```

를 볼 수 있다.

---

# 32. Network Packet Drop

예:

```text
RX dropped rapidly increasing
```

이면 sensor data loss와 관련될 수 있다.

---

# 33. Ping Monitoring

장치 간 basic connectivity:

```bash
ping <device-ip>
```

로 확인할 수 있다.

하지만:

```text
ping OK
≠
Application Healthy
```

임을 기억한다.

---

# 34. Sensor Metrics

Robot에서는 system resource보다 sensor health가 더 중요할 수 있다.

예:

```text
LiDAR frequency
IMU frequency
Camera FPS
Joint state frequency
```

---

# 35. ROS Topic Rate

예:

```bash
ros2 topic hz /imu
```

Expected:

```text
200 Hz
```

Actual:

```text
195 Hz
```

이면 정상 범위일 수 있다.

하지만:

```text
10 Hz
```

까지 떨어졌다면 문제다.

---

# 36. Topic Frequency Monitoring

예:

```text
/imu

Expected:
200 Hz

Warning:
< 180 Hz

Critical:
< 100 Hz
```

같은 health rule을 만들 수 있다.

---

# 37. LiDAR Monitoring

예:

```text
Expected:
10 Hz

Actual:
10 Hz
```

뿐 아니라:

```text
Point count
Timestamp
Packet drop
```

도 볼 수 있다.

---

# 38. Camera Monitoring

Camera:

```text
FPS
Dropped frames
Exposure
Image timestamp delay
```

등을 모니터링할 수 있다.

---

# 39. Timestamp Monitoring

Chapter 11과 연결된다.

예:

```text
Current System Time
-
Sensor Timestamp
=
Age
```

를 계산할 수 있다.

---

# 40. Data Age

예:

```text
Current:
10.500

Message timestamp:
10.100

Age:
400 ms
```

이면 최신 sensor data가 아니라
오래된 데이터를 처리하고 있을 수 있다.

---

# 41. Queue Buildup Monitoring

Queue가 쌓이면:

```text
Topic frequency는 정상
```

이어도 latency가 증가할 수 있다.

그래서:

```text
Message Age
Queue Size
```

도 중요한 metric이다.

---

# 42. FAST-LIO2 Metrics

FAST-LIO2에서는 예를 들어:

```text
Input LiDAR Rate
Input IMU Rate
Output Odometry Rate
Processing Time
Map Size
Residual
Dropped Frames
```

등을 metric으로 만들 수 있다.

---

# 43. SLAM Health

단순히 `/odometry`가 나온다고 정상은 아니다.

예:

```text
Pose jump
Large drift
Map corruption
```

이 발생할 수 있다.

따라서 quality metric도 필요하다.

---

# 44. Quality Metric

예:

```text
Residual
Covariance
Registration score
Innovation
Tracking status
```

등이 health indicator가 될 수 있다.

---

# 45. Navigation Metrics

예:

```text
Goal success rate
Planning latency
Controller latency
Costmap update rate
Recovery count
```

---

# 46. Application-Level Metric

좋은 observability는 system metric을 넘어서 application 의미를 포함한다.

예:

```text
CPU = 50%
```

보다:

```text
Localization valid = false
```

가 더 중요한 순간도 있다.

---

# 47. KPI

System 운영에서 핵심 metric을:

```text
KPI
```

로 정의할 수 있다.

예:

```text
Localization uptime
Navigation success
Average deployment success
Mean recovery time
```

---

# 48. Robot Dashboard

여러 metric을 시각적으로 보여주는 화면:

```text
Dashboard
```

이다.

예:

```text
Robot 001

Status: Healthy

CPU: 45%
GPU: 72%
RAM: 52%
Disk: 61%
Temp: 68°C

LiDAR: 10 Hz
IMU: 198 Hz
Odom: 100 Hz
```

---

# 49. Fleet Dashboard

여러 robot을 한 화면에서 볼 수 있다.

예:

```text
Robot 001   Healthy
Robot 002   Warning
Robot 003   Offline
Robot 004   Critical
```

---

# 50. Status Color만 보면 부족하다

```text
Green
Yellow
Red
```

만 보여주면 원인을 알기 어렵다.

Status와 함께:

```text
Reason
Timestamp
Relevant Metrics
```

를 보여주는 것이 좋다.

---

# 51. Alert

문제가 발생했을 때 operator에게 알림을 보내는 것을:

```text
Alert
```

이라고 한다.

예:

```text
Disk > 95%
LiDAR offline
Temperature critical
Localization lost
```

---

# 52. Alert Fatigue

Alert가 너무 많으면 사람은 결국 무시하게 된다.

```text
Warning
Warning
Warning
Warning
...
```

이를:

```text
Alert Fatigue
```

라고 한다.

---

# 53. 좋은 Alert

좋은 alert는:

```text
Actionable
```

해야 한다.

즉:

> 이 alert를 받으면 무엇을 해야 하는가?

가 명확해야 한다.

---

# 54. Severity

Alert를 level로 나눌 수 있다.

예:

```text
Info
Warning
Critical
```

---

# 55. Example

```text
Info
Software updated successfully

Warning
Disk > 80%

Critical
Localization lost while autonomous
```

---

# 56. Threshold Alert

가장 단순한 alert 방식:

```text
Temperature > 85°C
```

같은 threshold 기반이다.

---

# 57. Duration 조건

순간적으로 한 번 threshold를 넘었다고 바로 alert하면 noise가 많을 수 있다.

예:

```text
CPU > 90%
for 5 minutes
```

처럼 duration 조건을 추가할 수 있다.

---

# 58. Rate-of-Change Alert

값 자체보다 빠르게 변하는 것이 문제일 수 있다.

예:

```text
Disk usage:
50% → 80% in 5 min
```

log flooding이나 runaway recording을 의심할 수 있다.

---

# 59. Missing Data Alert

metric 자체가 안 들어오는 것도 문제다.

예:

```text
Robot heartbeat absent for 2 min
```

---

# 60. Alert Dependency

Robot이 offline이라면:

```text
LiDAR missing
IMU missing
CPU metric missing
```

alert가 수십 개 발생할 수 있다.

더 상위 원인:

```text
Robot Offline
```

하나로 묶는 것이 좋을 수 있다.

---

# 61. Root Cause

Observability의 중요한 목표는:

```text
Symptom
```

이 아니라:

```text
Root Cause
```

를 찾는 것이다.

예:

```text
FAST-LIO2 slow
```

증상.

실제 root cause:

```text
NVMe logging saturated memory bandwidth
```

일 수 있다.

---

# 62. Correlation

여러 metric을 같은 시간축에서 비교하면 원인을 찾기 쉽다.

예:

```text
10:30
CPU ↑
Temp ↑
Clock ↓
Odometry Rate ↓
```

이렇게 보면 thermal throttling을 의심할 수 있다.

---

# 63. Timestamp가 중요한 이유

모든 log/metric/event가 정확한 timestamp를 가져야
서로 연결할 수 있다.

Chapter 11의 time synchronization이 observability에도 중요하다.

---

# 64. Multi-Computer Robot

Xavier와 Orin이 따로 있다면:

```text
Xavier Logs
Orin Logs
LiDAR Events
Cloud Events
```

시간이 맞아야 하나의 incident를 재구성할 수 있다.

---

# 65. Structured Logging

Text만 자유롭게 쓰는 대신 구조화된 log를 사용할 수 있다.

예:

```json
{
  "event": "lidar_timeout",
  "sensor": "front_lidar",
  "severity": "error"
}
```

개념적으로 이런 구조다.

---

# 66. Structured Log의 장점

Machine이 쉽게:

```text
Filter
Search
Aggregate
Analyze
```

할 수 있다.

---

# 67. Log Context

좋은 log에는 다음이 있으면 유용하다.

```text
Timestamp
Robot ID
Process
Node
Severity
Event
Relevant values
```

---

# 68. Bad Log Example

```text
Error.
```

이건 거의 도움이 안 된다.

---

# 69. Better Log

```text
LiDAR timeout: no packet received for 500 ms
```

더 좋다.

---

# 70. Even Better

```text
robot=vision60-001
sensor=lidar_front
event=packet_timeout
duration_ms=500
```

처럼 context가 있으면 분석하기 쉽다.

---

# 71. Correlation ID

하나의 요청이나 작업에 unique ID를 붙일 수 있다.

예:

```text
request_id=abc123
```

여러 service의 log를 연결할 수 있다.

---

# 72. Trace ID

Distributed tracing에서는:

```text
Trace ID
Span ID
```

같은 identifier를 사용한다.

---

# 73. Span

Trace 안의 하나의 작업 구간:

```text
Span
```

이다.

예:

```text
Span 1: Camera Capture
Span 2: Preprocess
Span 3: Inference
Span 4: Publish
```

---

# 74. Latency Breakdown

Trace를 사용하면:

```text
Total latency = 50 ms
```

에서:

```text
Capture       5 ms
Preprocess   20 ms
Inference    15 ms
Publish      10 ms
```

처럼 분해할 수 있다.

---

# 75. Profiling vs Tracing

Profiling:

```text
한 process 내부에서 CPU time이 어디에 쓰였는가?
```

Tracing:

```text
하나의 요청이 여러 component를 어떻게 지나갔는가?
```

에 더 가깝다.

---

# 76. Metrics Cardinality

Monitoring system에서는 label 종류가 너무 많으면
데이터 양이 크게 증가할 수 있다.

예:

```text
robot_id
topic
sensor
process
```

는 유용하다.

하지만 매 요청마다 unique value를 metric label로 넣으면 비효율적일 수 있다.

---

# 77. Sampling

모든 trace/log를 항상 저장하면 비용이 클 수 있다.

그래서:

```text
Sampling
```

을 사용할 수 있다.

예:

```text
Normal trace:
1%

Error trace:
100%
```

---

# 78. Edge Logging

Robot이 offline일 수도 있으므로
일부 observability data를 local에 저장해야 한다.

```text
Robot
   │
   ├── local logs
   ├── local metrics buffer
   └── local events
```

---

# 79. Store-and-Forward

Network가 복구되면:

```text
Local Buffer
     ↓
Upload
     ↓
Cloud
```

한다.

Chapter 17과 연결된다.

---

# 80. Offline Retention

Local storage는 무한하지 않다.

따라서:

```text
Logs 7 days
Metrics 24 hours
Critical events keep longer
```

같은 retention policy가 필요하다.

---

# 81. Observability Data도 Disk를 채울 수 있다

아이러니하게도 monitoring을 너무 많이 하면:

```text
Disk Full
```

을 만들 수 있다.

따라서 Chapter 13의 storage 관리와 연결된다.

---

# 82. Monitoring Overhead

Observability 자체도:

```text
CPU
RAM
Disk
Network
```

를 사용한다.

즉:

```text
Measurement
affects system
```

가능성이 있다.

---

# 83. 너무 높은 Metric Frequency

예:

```text
CPU telemetry 1000 Hz
```

는 대부분 필요 없다.

Metric 성격에 따라 frequency를 정한다.

---

# 84. Metric Frequency Example

```text
CPU
1 Hz

Temperature
1 Hz

Disk
0.1 Hz

Software Version
on change

Critical Event
immediately
```

처럼 할 수 있다.

---

# 85. Push vs Pull

Monitoring data 수집에는 두 방식이 있다.

```text
Push
Pull
```

---

# 86. Push

Robot이 monitoring server로 직접 보낸다.

```text
Robot
  │
  │ metrics
  ▼
Server
```

IoT/field robot에 잘 맞을 수 있다.

---

# 87. Pull

Monitoring server가 robot의 endpoint를 주기적으로 읽는다.

```text
Server
  │
  │ scrape
  ▼
Robot
```

Datacenter 환경에서 흔하다.

---

# 88. Robot에서는 Push가 유리할 수 있다

Field robot은:

```text
NAT
Firewall
Unstable Network
Mobile connection
```

뒤에 있을 수 있다.

그래서 robot이 outbound connection으로 보내는 방식이 관리하기 쉬울 수 있다.

---

# 89. Prometheus 개념

Prometheus는 time-series metrics monitoring에서 자주 사용되는 system이다.

전형적인 구조:

```text
Exporter
   │
   ▼
Prometheus
   │
   ▼
Dashboard / Alert
```

---

# 90. Exporter

System 상태를 metric 형태로 노출하는 component다.

예:

```text
CPU
Memory
Disk
Network
```

---

# 91. Grafana 개념

Grafana는 metrics/log data를 dashboard로 시각화하는 데 많이 사용된다.

예:

```text
CPU graph
Temperature graph
Robot fleet table
```

---

# 92. Loki 개념

Loki 같은 system은 log aggregation에 사용할 수 있다.

개념:

```text
Robot Logs
   ↓
Log Backend
   ↓
Search / Dashboard
```

---

# 93. OpenTelemetry

OpenTelemetry는:

```text
Metrics
Logs
Traces
```

를 수집하기 위한 표준화된 observability framework다.

---

# 94. Robot에 꼭 이런 tool을 써야 하나?

아니다.

중요한 것은 tool 이름이 아니라:

```text
무엇을 측정할 것인가?
어떻게 저장할 것인가?
어떻게 alert할 것인가?
```

이다.

---

# 95. ROS Diagnostics

ROS ecosystem에는:

```text
diagnostic_msgs
```

등을 이용해 component health를 표현하는 방식도 있다.

예:

```text
OK
WARN
ERROR
STALE
```

상태를 나타낼 수 있다.

---

# 96. Diagnostic Aggregator

여러 sensor/node diagnostics를 모아
robot 전체 health를 구성할 수 있다.

```text
LiDAR OK
IMU OK
Camera WARN
Battery OK
      │
      ▼
Robot Health
```

---

# 97. Heartbeat Topic

Custom heartbeat topic을 만들 수도 있다.

예:

```text
/fastlio/heartbeat
/navigation/heartbeat
```

하지만 너무 많은 custom mechanism을 만들기보다
표준 diagnostic framework를 검토하는 것이 좋다.

---

# 98. ROS Node Monitoring

확인:

```bash
ros2 node list
```

Fleet system에서는 node list를 주기적으로 수집할 수도 있다.

하지만 node 존재만으로 health를 판단하면 부족하다.

---

# 99. Topic Connectivity

예:

```text
LiDAR publisher exists
FAST-LIO2 subscriber exists
```

도 health indicator가 될 수 있다.

---

# 100. QoS Monitoring

Topic은 존재하지만 QoS mismatch 때문에 data가 안 흐를 수 있다.

따라서:

```bash
ros2 topic info /topic --verbose
```

에서 QoS를 확인하는 것도 debugging에 중요하다.

---

# 101. DDS Metrics

더 깊게 보면:

```text
Discovery
Packet loss
Matched endpoints
Transport statistics
```

같은 DDS-level 정보도 observability 대상이 될 수 있다.

---

# 102. Application State

Robot application은 단순 healthy/unhealthy보다
현재 mode를 알려주는 것이 좋다.

예:

```text
BOOTING
IDLE
MAPPING
NAVIGATING
PAUSED
ERROR
ESTOP
```

---

# 103. State Machine Monitoring

현재 application state를 dashboard에 표시하면
operator가 robot 상황을 빠르게 이해할 수 있다.

---

# 104. Error Code

Error를 text만 보내는 대신
고유한 code를 정의할 수도 있다.

예:

```text
E101
LiDAR timeout

E205
Localization lost
```

---

# 105. Error Code 장점

```text
Dashboard
Documentation
Alert automation
Statistics
```

와 연결하기 쉽다.

---

# 106. Incident

여러 warning/error가 하나의 큰 문제에서 발생할 수 있다.

예:

```text
Network switch failure
      ↓
LiDAR lost
      ↓
FAST-LIO2 lost
      ↓
Navigation stops
```

이 전체 사건을:

```text
Incident
```

로 묶어 분석할 수 있다.

---

# 107. Incident Timeline

```text
14:00:00 Network packet loss rises
14:00:02 LiDAR rate drops
14:00:03 Localization warning
14:00:04 Navigation stops
```

이렇게 timeline을 만들면 root cause를 찾기 쉽다.

---

# 108. MTTR

MTTR은 문맥에 따라:

```text
Mean Time To Repair
```

또는 recovery 관련 의미로 사용된다.

운영에서는:

> 문제가 발생한 후 정상 상태로 복구하는 데 얼마나 걸리는가?

를 측정하는 데 사용할 수 있다.

---

# 109. Reliability와 Observability

좋은 observability는 failure 자체를 없애지는 않는다.

하지만:

```text
Failure detection time ↓
Root cause analysis time ↓
Recovery time ↓
```

에 도움을 준다.

---

# 110. SLI

SLI:

```text
Service Level Indicator
```

이다.

System quality를 측정하는 실제 metric이다.

예:

```text
Localization uptime
Navigation success rate
```

---

# 111. SLO

SLO:

```text
Service Level Objective
```

이다.

원하는 목표 수준이다.

예:

```text
Localization valid
>= 99.5% of mission time
```

---

# 112. Robot SLO Example

예:

```text
LiDAR availability
> 99%

Odometry rate
> 95 Hz during mission

Temperature
< threshold for 99% of runtime
```

실제 기준은 robot 요구사항에 맞춰 정한다.

---

# 113. Not Every Metric Needs an Alert

Metric은 많이 수집해도 되지만
모든 metric에 alert를 걸 필요는 없다.

예:

```text
CPU Usage
→ dashboard metric

Localization Lost
→ immediate alert
```

처럼 중요도를 구분한다.

---

# 114. Golden Signals

Cloud system에서는 흔히:

```text
Latency
Traffic
Errors
Saturation
```

같은 핵심 signal을 본다.

Robot에서도 비슷하게 적용할 수 있다.

---

# 115. Robot Golden Signals

예:

```text
Latency
→ perception / control latency

Traffic
→ sensor/network rate

Errors
→ sensor failure / localization failure

Saturation
→ CPU/GPU/RAM/Disk utilization
```

---

# 116. Vision60 Suggested Metrics

예:

```text
System

CPU %
GPU %
RAM %
Disk %
CPU Temp
GPU Temp

Network

RX/TX Mbps
Packet drop
Latency

Sensors

LiDAR Hz
IMU Hz
Camera FPS
Joint State Hz

SLAM

Odometry Hz
Processing latency
Tracking state
Map size

Robot

Battery
Mode
E-stop
Current gait
```

---

# 117. Alert Examples

예:

```text
Critical
E-stop triggered

Critical
Localization invalid while autonomous

Critical
Jetson temperature above safe threshold

Warning
Disk > 85%

Warning
LiDAR frequency below expected rate
```

---

# 118. Alert Context

Alert에는 가능하면:

```text
Robot ID
Time
Current Mode
Current Version
Related Metrics
```

를 포함한다.

---

# 119. Version 정보가 왜 필요할까?

예:

```text
Robot 001
Failure

Software v1.7
```

Robot 002:

```text
Software v1.6
Healthy
```

이면 software regression을 의심할 수 있다.

Chapter 17과 연결된다.

---

# 120. Config 정보도 필요하다

같은 software인데 특정 site에서만 문제라면:

```text
Site Config
Map
Calibration
```

차이를 볼 수 있어야 한다.

---

# 121. Observability Metadata

모든 monitoring data에 다음 metadata가 있으면 유용하다.

```text
Robot ID
Site
Hardware Model
Software Version
Config Version
Timestamp
```

---

# 122. Dashboard 계층

Dashboard를 여러 level로 나눌 수 있다.

```text
Fleet Overview
     ↓
Robot Detail
     ↓
Component Detail
```

---

# 123. Fleet Overview

예:

```text
Total robots: 50

Healthy: 44
Warning: 4
Critical: 1
Offline: 1
```

---

# 124. Robot Detail

예:

```text
Vision60-007

Mode:
NAVIGATING

CPU:
62%

GPU:
75%

LiDAR:
10 Hz

Odometry:
98 Hz
```

---

# 125. Component Detail

FAST-LIO2:

```text
Input LiDAR
Input IMU
Processing latency
Output rate
Residual
Memory
```

등을 본다.

---

# 126. Drill Down

Operator가:

```text
Fleet Warning
```

을 클릭하면:

```text
Robot 007
```

그리고:

```text
LiDAR
```

까지 좁혀갈 수 있는 구조가 좋다.

---

# 127. Observability Pipeline

전체 구조:

```text
                    Robot

        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Logs         Metrics       Events
        │            │            │
        └────────────┼────────────┘
                     ▼
              Collection Agent
                     │
                     ▼
             Local Buffer
                     │
                     ▼
               Network
                     │
                     ▼
              Monitoring Backend
                     │
            ┌────────┴────────┐
            ▼                 ▼
        Dashboard           Alert
```

---

# 128. Edge Agent

Robot에서 observability data를 수집하는 agent를 둘 수 있다.

예:

```text
CPU
GPU
ROS health
Disk
Network
```

를 하나로 모아 cloud로 보낸다.

---

# 129. Agent Failure

Monitoring agent 자체가 죽을 수도 있다.

그래서:

```text
No telemetry
```

가:

```text
Robot dead
```

인지:

```text
Monitoring agent dead
```

인지 구분해야 한다.

---

# 130. Heartbeat Layer

예:

```text
Robot Agent Heartbeat
ROS Health
Sensor Health
```

를 별도로 구분하면 진단에 도움이 된다.

---

# 131. Out-of-Band Monitoring

가능하다면 robot main application과 독립된 monitoring path를 둘 수도 있다.

예:

```text
Autonomy crashed
```

해도:

```text
Monitoring agent alive
```

라면 원격으로 상태를 확인할 수 있다.

---

# 132. Crash Dump

Program crash 시:

```text
Stack trace
Core dump
Crash log
```

를 보존하면 debugging에 도움이 된다.

---

# 133. Core Dump

Linux process가 crash했을 때 memory/process 상태를 기록한 file을:

```text
Core Dump
```

라고 한다.

C/C++ segfault 분석에 매우 유용하다.

---

# 134. Core Dump는 클 수 있다

Large process는 core dump가 매우 커질 수 있다.

따라서 storage policy가 필요하다.

---

# 135. Anomaly Detection

단순 threshold가 아니라
평소 pattern과 다른 상태를 감지할 수도 있다.

예:

```text
평상시 CPU = 40~60%

갑자기 85%
```

같은 변화.

---

# 136. Trend Detection

예:

```text
Disk:
60%
65%
70%
75%
80%
```

지금은 critical이 아니어도
곧 문제가 생길 것을 예상할 수 있다.

---

# 137. Predictive Maintenance

장기간 metric을 분석하면:

```text
Fan degradation
SSD wear
Battery degradation
Sensor instability
```

등을 미리 예측하는 데 사용할 수 있다.

---

# 138. Observability와 Privacy

Camera, audio, location 같은 데이터는 민감할 수 있다.

Monitoring을 위해 필요 이상으로 raw data를 cloud에 보내지 않는 것이 좋다.

---

# 139. Minimum Necessary Data

예:

```text
Camera raw video
```

대신:

```text
Camera FPS
Camera healthy=true
```

만 보내도 health monitoring에는 충분할 수 있다.

---

# 140. Security

Observability data에도 민감한 정보가 들어갈 수 있다.

예:

```text
Robot location
Network info
Software version
Logs
```

Chapter 16의:

```text
Authentication
Encryption
Authorization
```

을 적용해야 한다.

---

# 141. Alert Channel

Critical alert를:

```text
Email
Slack
SMS
Pager
Dashboard
```

등으로 보낼 수 있다.

실제 운영 방식에 맞춰 선택한다.

---

# 142. Alert Escalation

예:

```text
Warning
→ Dashboard

Critical
→ Slack

Critical for 10 min
→ Operator call
```

처럼 escalation policy를 둘 수 있다.

---

# 143. Maintenance Window

Update나 maintenance 중에는 일부 alert가 예상된다.

예:

```text
Robot intentionally offline
```

인데 offline critical alert가 계속 오면 불필요하다.

Maintenance state를 monitoring에 반영할 수 있다.

---

# 144. Silencing

Known maintenance 동안 특정 alert를 임시로 suppress하는 것을:

```text
Silencing
```

이라고 할 수 있다.

하지만 실제 critical issue까지 숨기지 않도록 주의해야 한다.

---

# 145. Postmortem

큰 장애 이후:

```text
What happened?
Why?
How detected?
How recovered?
How prevent recurrence?
```

를 문서화하는 과정을 postmortem으로 볼 수 있다.

---

# 146. Observability가 Postmortem을 가능하게 한다

좋은:

```text
Logs
Metrics
Events
Versions
```

가 있으면 사고를 재구성할 수 있다.

없으면:

```text
"현장에서 뭔가 이상했습니다."
```

수준에 머물게 된다.

---

# 147. Mini Practice 1

Vision60에서 monitoring해야 할 metric을 직접 분류한다.

```text
System
Sensor
ROS
SLAM
Navigation
Safety
```

각 category별로 최소 3개씩 정한다.

---

# 148. Mini Practice 2

FAST-LIO2 health rule을 만든다.

예:

```text
Healthy

LiDAR >= 9 Hz
IMU >= 180 Hz
Odometry >= 90 Hz
Tracking valid
```

---

# 149. Mini Practice 3

다음 상황을 생각한다.

```text
CPU = 95%
Temperature = 88°C
Odometry = 60 Hz
```

질문:

```text
어떤 metric을 더 확인해야 하는가?
```

예:

```text
Clock
Thermal throttling
LiDAR rate
FAST-LIO2 processing latency
```

---

# 150. Mini Practice 4

Disk가:

```text
70%
72%
75%
82%
90%
```

로 빠르게 증가한다.

질문:

```text
단순 threshold alert 외에 무엇을 볼 수 있는가?
```

답:

```text
Rate of change
Which process is writing?
rosbag?
Docker logs?
```

---

# 151. Mini Practice 5

Robot:

```text
Online
```

인데 FAST-LIO2가 죽었다.

이 상황에 필요한 monitoring layer:

```text
Fleet connectivity
Process health
ROS node health
Topic health
Application health
```

를 구분한다.

---

# 152. Mini Practice 6

다음 incident timeline을 만든다.

```text
14:00 LiDAR packet loss starts
14:01 LiDAR rate drops
14:02 FAST-LIO2 warning
14:03 Localization invalid
14:04 Navigation stops
```

질문:

```text
Root cause 후보는 어디인가?
```

---

# 153. 반드시 구분할 것

```text
Monitoring
≠
Observability

Log
≠
Metric

Metric
≠
Event

Alive
≠
Healthy

Liveness
≠
Readiness

Online
≠
Healthy

Profiling
≠
Tracing

Alert
≠
Metric

Threshold
≠
Root Cause

Telemetry
≠
Raw Sensor Data
```

---

# 154. Observability Mental Model

```text
System
  │
  ├── What happened?
  │      ↓
  │     Logs
  │
  ├── How much?
  │      ↓
  │    Metrics
  │
  ├── Where did time go?
  │      ↓
  │    Traces
  │
  └── Is it healthy?
         ↓
       Health
```

---

# 155. Robot Monitoring Mental Model

```text
                       Robot

         Hardware
            │
            ▼
        Linux / Jetson
            │
            ▼
           ROS 2
            │
            ▼
        Application
            │
            ▼
          Mission

각 Layer
   │
   ├── Logs
   ├── Metrics
   ├── Events
   └── Health
   │
   ▼
Monitoring System
```

---

# 156. Chapter 17과 Chapter 18 연결

Chapter 17:

```text
What software should this robot run?
```

Chapter 18:

```text
Is that software actually running correctly?
```

이다.

즉:

```text
Deploy
   ↓
Observe
   ↓
Evaluate
   ↓
Rollback / Continue
```

가 하나의 운영 loop가 된다.

---

# 157. Chapter 18 핵심

좋은 robot observability system은:

```text
Robot is online
```

만 알려주는 것이 아니다.

다음 질문에 답할 수 있어야 한다.

```text
What is failing?

When did it start?

Which layer is affected?

How severe is it?

Which software version is running?

Is the issue getting worse?

What should the operator do?
```

---

# Next Chapter

## Chapter 19. Reliability & Fault Tolerance

마지막 Chapter에서는:

```text
Failure
Fault
Error
Redundancy
Watchdog
Retry
Timeout
Graceful Degradation
Fail-Safe
Fail-Operational
Recovery
Redundant Sensors
Single Point of Failure
```

를 다룬다.

핵심 질문은:

```text
"문제가 생기지 않게 하는 방법"
```

을 넘어:

```text
"문제는 언젠가 생긴다고 가정했을 때,
Robot이 어떻게 안전하게 계속 동작하거나 복구할 것인가?"
```

이다.

Chapter 19까지 끝나면 Edge Computing 트랙을:

```text
Hardware
→ OS
→ Network
→ Runtime
→ Deployment
→ Monitoring
→ Reliability
```

까지 완성하게 된다.