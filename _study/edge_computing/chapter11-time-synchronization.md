---
title: "Chapter 11. Time Synchronization & Sensor Timing"
importance: 12
---

> **Goal:** 로봇에서 여러 센서와 여러 컴퓨터의 timestamp를 어떻게 맞추는지 이해한다.
> 특히 LiDAR, IMU, Camera, Joint Encoder를 fusion할 때 왜 시간 정렬이 중요한지,
> NTP, PTP, hardware timestamp, ROS 2 timestamp가 어떻게 연결되는지 이해하는 것이 목표다.

---

# 1. 로봇에서 시간은 왜 중요한가?

Sensor fusion은 단순히 여러 센서 데이터를 동시에 사용하는 것이 아니다.

실제로는:

```text
같은 시점의 데이터를
정확히 맞춰서 사용
```

해야 한다.

예를 들어:

```text
LiDAR scan = 10:00:00.100

IMU       = 10:00:00.100
```

이면 같은 시점의 motion을 설명한다.

하지만:

```text
LiDAR = 10:00:00.100

IMU   = 10:00:00.180
```

이면 80 ms 차이가 난다.

로봇이 빠르게 움직이는 중이라면 이 차이는 매우 큰 오차를 만들 수 있다.

---

# 2. Timestamp란?

Timestamp는:

> 특정 데이터가 어느 시점에 발생했는지 나타내는 시간 정보

다.

예:

```text
2026-09-05 21:10:32.123456
```

또는 ROS 2에서는:

```text
sec
nanosec
```

형태로 표현될 수 있다.

예:

```text
sec: 1788610232
nanosec: 123456789
```

---

# 3. Sensor Data와 Timestamp

센서 message는 일반적으로:

```text
Sensor Data
+
Timestamp
```

로 생각해야 한다.

예:

```text
IMU

angular_velocity
linear_acceleration
timestamp
```

```text
LiDAR

point cloud
timestamp
```

즉 데이터 값만 맞는 것이 아니라
시간도 정확해야 한다.

---

# 4. ROS 2 Header

ROS 2 sensor message에는 자주:

```text
std_msgs/Header
```

가 들어간다.

예:

```text
sensor_msgs/msg/Imu

header
angular_velocity
linear_acceleration
...
```

Header에는 보통:

```text
stamp
frame_id
```

가 있다.

---

# 5. `header.stamp`

예:

```text
msg.header.stamp
```

는 이 sensor message의 시간 정보를 나타낸다.

FAST-LIO2 같은 sensor fusion algorithm은
이 timestamp를 이용해 LiDAR와 IMU를 정렬할 수 있다.

---

# 6. `frame_id`

Header의 또 다른 중요한 값:

```text
frame_id
```

는 이 데이터가 어느 coordinate frame 기준인지 나타낸다.

예:

```text
frame_id = imu_link
```

즉:

```text
Timestamp
→ 언제?

Frame ID
→ 어디 기준?
```

이다.

---

# 7. Sensor Fusion에서 시간 오차

예를 들어 로봇이 회전하고 있다고 하자.

```text
t = 0.00 s
orientation = 0 deg

t = 0.10 s
orientation = 20 deg
```

10 Hz 수준으로 빠르게 움직이면 100 ms 동안 20도나 회전할 수 있다.

그런데 LiDAR와 IMU timestamp가 100 ms 어긋나면:

```text
LiDAR는 과거 자세
IMU는 현재 자세
```

를 섞게 된다.

결과:

```text
Map distortion
Pose error
Deskew failure
```

가 발생할 수 있다.

---

# 8. LiDAR Deskew와 Timestamp

LiDAR 한 scan은 순간적으로 생성되는 것이 아니다.

예:

```text
Scan Start
t = 0.000

Point 1
Point 2
Point 3
...

Scan End
t = 0.100
```

즉 하나의 scan 안에서도 point마다 측정 시간이 다를 수 있다.

---

# 9. Motion Distortion

로봇이 scan 도중 움직이면 point cloud가 찌그러질 수 있다.

예:

```text
Robot Moving

Point A measured at t=0.00
Point B measured at t=0.05
Point C measured at t=0.10
```

이 point들을 마치 같은 시점에 찍었다고 가정하면 distortion이 생긴다.

---

# 10. Deskew

Deskew는 각 point의 측정 시점을 고려해서
motion을 보정하는 과정이다.

구조:

```text
LiDAR Point Timestamp
        +
IMU Motion
        ↓
Deskew
        ↓
Corrected Point Cloud
```

FAST-LIO2에서 중요한 과정 중 하나다.

---

# 11. IMU가 왜 시간 해상도가 높은가?

IMU는 보통 LiDAR보다 훨씬 높은 frequency로 측정한다.

예:

```text
LiDAR
10 Hz

IMU
200 Hz
```

그러면 LiDAR scan 사이에 많은 IMU measurement가 존재한다.

```text
LiDAR Scan
|----------------|

IMU
| | | | | | | | |
```

이 IMU를 이용해서 scan 내부 motion을 추정한다.

---

# 12. Frequency와 Period

Frequency가:

```text
200 Hz
```

이면 초당 200번 측정한다.

Period는:

```text
1 / 200
=
0.005 s
=
5 ms
```

이다.

즉 IMU measurement가 약 5 ms마다 들어온다.

---

# 13. 여러 Sensor Clock

문제는 모든 sensor가 같은 clock을 사용하지 않을 수 있다는 점이다.

예:

```text
LiDAR Clock
Camera Clock
IMU Clock
Jetson Clock
Xavier Clock
MCU Clock
```

각각 독립적인 oscillator를 가질 수 있다.

---

# 14. Clock Drift

두 clock은 처음에는 같은 시간을 가리켜도
시간이 지나면서 조금씩 차이가 날 수 있다.

이를:

```text
Clock Drift
```

라고 한다.

예:

```text
Start

Clock A = 0.000
Clock B = 0.000

1 hour later

Clock A = 3600.000
Clock B = 3600.120
```

120 ms 차이가 생길 수도 있다.

---

# 15. Clock Offset

두 clock 사이의 현재 시간 차이를:

```text
Clock Offset
```

이라고 한다.

예:

```text
Jetson clock = 10:00:00.000
LiDAR clock  = 10:00:00.050
```

Offset:

```text
+50 ms
```

---

# 16. Offset와 Drift 차이

```text
Offset
→ 지금 얼마나 차이 나는가?

Drift
→ 시간이 지나면서 차이가 얼마나 변하는가?
```

둘 다 중요하다.

---

# 17. System Clock

Linux computer는 system clock을 가진다.

확인:

```bash
date
```

더 자세히:

```bash
timedatectl
```

을 사용할 수 있다.

---

# 18. `timedatectl`

예:

```bash
timedatectl
```

확인할 수 있는 것:

```text
Local time
Universal time
RTC time
Time zone
NTP status
```

---

# 19. RTC

RTC는:

**Real-Time Clock**

이다.

Computer가 꺼져 있어도 시간을 유지하는 hardware clock이다.

Boot 시 system time을 초기화하는 데 사용할 수 있다.

---

# 20. System Clock vs Hardware Clock

단순화하면:

```text
Hardware RTC
      ↓
Boot
      ↓
Linux System Clock
```

그리고 system clock은 NTP/PTP 등에 의해 보정될 수 있다.

---

# 21. UTC

분산 시스템에서는 local timezone보다
UTC를 기준으로 관리하는 것이 일반적으로 편리하다.

예:

```text
Korea Local Time
UTC+9
```

하지만 sensor fusion에서 중요한 것은 timezone 자체보다
각 장치의 timestamp가 동일한 기준에 맞는지다.

---

# 22. NTP

NTP는:

**Network Time Protocol**

이다.

Network를 통해 computer clock을 동기화한다.

구조:

```text
Time Server
     │
     ▼
   Network
     │
     ├── Xavier
     ├── Orin
     └── Laptop
```

---

# 23. NTP가 필요한 이유

예를 들어:

```text
Xavier
FAST-LIO2

Orin
Camera AI
```

가 서로 다른 computer라면 clock이 다를 수 있다.

NTP를 사용하면 이 clock 차이를 줄일 수 있다.

---

# 24. NTP는 어느 정도 정확한가?

NTP 정확도는:

```text
Network latency
Network jitter
Server quality
Configuration
```

등에 따라 달라진다.

일반 computer synchronization에는 충분한 경우가 많지만
고정밀 sensor fusion에서는 더 높은 정확도가 필요할 수 있다.

---

# 25. PTP

PTP는:

**Precision Time Protocol**

이다.

NTP보다 더 높은 정밀도의 clock synchronization을 목표로 한다.

Industrial automation, robotics, telecom 등에서 사용된다.

---

# 26. NTP vs PTP

단순 비교:

| NTP | PTP |
|---|---|
| 일반 network time sync | 고정밀 time sync |
| 설정 비교적 단순 | 설정 복잡 |
| 일반 server sync | 산업/센서 sync |
| software timestamp 가능 | hardware timestamp 활용 가능 |

---

# 27. PTP Clock Structure

PTP에서는 일반적으로:

```text
Grandmaster Clock
       │
       ▼
    Network
       │
       ├── Device A
       ├── Device B
       └── Device C
```

구조를 사용한다.

---

# 28. Grandmaster

Grandmaster는 전체 PTP network의 기준 clock 역할을 한다.

다른 device들은 이 clock에 맞춘다.

```text
Grandmaster
    │
    ▼
Jetson
LiDAR
Camera
```

---

# 29. Hardware Timestamp

Network packet timestamp를 software에서 찍는 대신
NIC hardware에서 직접 기록할 수 있다.

```text
Packet arrives
      │
      ▼
NIC Hardware
      │
Timestamp
      │
      ▼
Kernel
```

이를 hardware timestamping이라고 한다.

---

# 30. Software Timestamp

Software timestamp는 packet이 kernel이나 application layer에 도착한 뒤
시간을 기록할 수 있다.

```text
Physical Arrival
     ↓
Driver
     ↓
Kernel
     ↓
Application
     ↓
Timestamp
```

중간 processing delay가 포함될 수 있다.

---

# 31. Hardware Timestamp 장점

Hardware timestamp는 실제 physical packet arrival 시점과 더 가까운 시간을 기록할 수 있다.

따라서:

```text
Network jitter 영향 감소
Software scheduling 영향 감소
```

효과가 있다.

---

# 32. LiDAR Hardware Synchronization

고급 LiDAR는 다음 입력을 지원할 수 있다.

```text
PTP
PPS
GPS
Trigger
```

이를 통해 sensor clock을 외부 기준에 맞출 수 있다.

---

# 33. PPS

PPS는:

**Pulse Per Second**

이다.

정확히 1초마다 pulse signal을 보낸다.

예:

```text
GPS Receiver
      │
      │ PPS
      ▼
LiDAR / IMU
```

정확한 초 경계를 맞추는 데 사용할 수 있다.

---

# 34. GPS Time Synchronization

GPS receiver는 매우 정확한 time source가 될 수 있다.

예:

```text
GPS
 │
 ├── Time Message
 └── PPS
      │
      ▼
Sensor
```

야외 autonomous system에서 많이 사용된다.

---

# 35. Trigger Synchronization

Camera 여러 대를 동시에 촬영하려면
hardware trigger를 사용할 수 있다.

```text
Trigger Generator
       │
       ├── Camera A
       ├── Camera B
       └── Camera C
```

같은 signal로 exposure timing을 맞춘다.

---

# 36. Software Sync vs Hardware Sync

## Software Sync

```text
Timestamp 비교 후
가장 가까운 데이터를 매칭
```

장점:

```text
구현 쉬움
추가 wiring 적음
```

단점:

```text
정확도 제한
Clock drift 영향
```

---

## Hardware Sync

```text
같은 clock / trigger 사용
```

장점:

```text
정확한 synchronization
```

단점:

```text
Hardware 구성 복잡
Sensor support 필요
```

---

# 37. Exact Synchronization

두 message timestamp가 정확히 같을 때만 pair를 만든다고 하자.

```text
LiDAR 10.000
IMU   10.000
```

이런 방식은 strict하다.

실제 sensor frequency가 다르면 exact match가 어려울 수 있다.

---

# 38. Approximate Synchronization

가까운 timestamp끼리 매칭하는 방법도 있다.

예:

```text
Camera
10.000

LiDAR
10.006
```

6 ms 차이 안에서 acceptable하다고 판단할 수 있다.

---

# 39. ROS Message Filters

ROS ecosystem에서는 여러 message를 timestamp 기준으로 동기화하는 helper를 사용할 수 있다.

예:

```text
Camera
    \
     \
      Synchronizer
     /
LiDAR
```

Exact / Approximate sync concept을 사용할 수 있다.

---

# 40. FAST-LIO2의 Sync

FAST-LIO2에서는 LiDAR와 IMU data를 buffer에 저장한 뒤
시간 기준으로 묶어서 processing한다.

개념:

```text
LiDAR Buffer
      │
      ├── scan t0
      ├── scan t1
      └── scan t2

IMU Buffer
      │
      ├── many IMU measurements
      └── ...

        ↓

sync_packages()

        ↓

LiDAR scan
+
corresponding IMU measurements
```

---

# 41. `sync_packages`

이름 그대로:

```text
sync
=
synchronize

packages
=
sensor data bundles
```

라고 이해할 수 있다.

FAST-LIO2에서 LiDAR scan과 해당 시간 구간의 IMU를 맞추는 역할을 한다.

---

# 42. Timestamp가 이상하면 생기는 현상

예:

```text
LiDAR time jumps backward
```

또는:

```text
IMU timestamp ahead of LiDAR
```

등이 발생하면:

```text
Buffer 문제
Sync 실패
Deskew 이상
Pose jump
Map distortion
```

가 생길 수 있다.

---

# 43. Monotonic Time

일부 algorithm에서는 시간이 항상 앞으로 증가해야 한다.

```text
1.0
1.1
1.2
1.3
```

이런 clock을 monotonic하게 본다.

---

# 44. Wall Clock

사람이 보는 실제 날짜/시간:

```text
2026-09-05 21:00
```

같은 clock이다.

NTP correction 등으로 시간이 조금 조정될 수 있다.

---

# 45. Monotonic Clock과 Wall Clock 차이

Wall clock은:

```text
NTP correction
manual change
timezone
```

등의 영향을 받을 수 있다.

Monotonic clock은 elapsed time 계산에 더 적합하다.

---

# 46. ROS Time

ROS에서는 clock source를 추상화해서 사용할 수 있다.

예:

```text
System Time
ROS Time
Steady Time
```

등을 구분할 수 있다.

---

# 47. Simulation Time

Isaac Sim이나 Gazebo 같은 simulator에서는 실제 wall clock 대신:

```text
Simulation Time
```

을 사용할 수 있다.

ROS 2 parameter:

```text
use_sim_time
```

를 사용할 수 있다.

---

# 48. `/clock`

Simulation에서는 `/clock` topic을 통해 simulated time을 제공할 수 있다.

```text
Simulator
   │
   │ /clock
   ▼
ROS 2 Nodes
```

---

# 49. `use_sim_time`

Node가:

```text
use_sim_time=true
```

이면 system clock 대신 simulation clock을 사용할 수 있다.

---

# 50. Simulation에서 시간 문제가 중요한 이유

Simulator가 pause되면:

```text
Simulation Time stops
```

할 수 있다.

실제 wall clock은 계속 흐른다.

따라서:

```text
sim time
≠
wall clock
```

이다.

---

# 51. Bag Replay와 Time

rosbag replay에서도 recorded timestamp를 기준으로
과거 sensor data를 재생할 수 있다.

```text
Recorded Data
      │
      ▼
Replay
      │
      ▼
ROS Nodes
```

시간을 실제보다 빠르게/느리게 재생할 수도 있다.

---

# 52. Timestamp Source

Sensor message timestamp가 어디에서 생성되는지 반드시 확인해야 한다.

가능한 source:

```text
Sensor hardware
Driver receipt time
Kernel timestamp
Jetson system clock
MCU clock
```

---

# 53. Receipt Time

Driver가 message를 받은 순간:

```text
now()
```

를 timestamp로 넣는 경우가 있을 수 있다.

하지만 이는 실제 measurement time과 다르다.

예:

```text
Sensor measures at 10.000
Network delay
Driver receives at 10.010
```

Receipt timestamp:

```text
10.010
```

measurement timestamp:

```text
10.000
```

---

# 54. 왜 Measurement Time이 더 중요할까?

Sensor fusion에서는:

> 언제 packet을 받았는가?

보다:

> 실제 물리량을 언제 측정했는가?

가 더 중요하다.

---

# 55. LiDAR Packet Time

LiDAR는 하나의 full point cloud가 한 packet으로 오는 것이 아닐 수 있다.

```text
Packet 1
Packet 2
Packet 3
...
```

각 packet과 point에 timing information이 있을 수 있다.

Driver가 이를 이용해 full scan timestamp를 계산한다.

---

# 56. Camera Exposure Time

Camera에서도 timestamp가 단순히 "image message 생성 시각"만 의미하는 것이 아닐 수 있다.

실제로 중요한 것은:

```text
Exposure Start
Exposure End
Frame Readout
Driver Arrival
```

등이다.

Rolling shutter camera에서는 특히 더 중요하다.

---

# 57. Rolling Shutter

Rolling shutter camera는 image 전체를 같은 순간에 촬영하지 않는다.

위쪽 row와 아래쪽 row의 exposure time이 다를 수 있다.

```text
Top row    t=0.000
Middle     t=0.005
Bottom     t=0.010
```

빠른 motion에서는 image distortion이 생길 수 있다.

---

# 58. Global Shutter

Global shutter는 전체 sensor가 거의 같은 시점에 exposure된다.

빠른 robot motion에서 geometry 측면에서 유리할 수 있다.

---

# 59. Joint Encoder Time

Quadruped에서 leg odometry를 사용한다면 joint encoder timestamp도 중요하다.

예:

```text
Joint Angle q
Joint Velocity qdot
Foot Contact
```

이 data가 IMU와 같은 시점이어야 body motion을 정확히 추정할 수 있다.

---

# 60. Contact Timing

사족보행에서는 foot contact state 변화가 빠르다.

예:

```text
Swing
   ↓
Touchdown
   ↓
Stance
```

contact timestamp가 늦으면
실제로는 stance인데 estimator는 아직 swing으로 판단할 수 있다.

---

# 61. Gait-aware Estimation과 Timing

구조:

```text
IMU
Joint Encoder
Foot Contact
LiDAR
     │
     ▼
Time Synchronization
     │
     ▼
State Estimator
```

Quadruped SLAM에서는 sensor 종류가 많아질수록 timing이 더 중요해진다.

---

# 62. Multi-Computer Sensor Fusion

예:

```text
Xavier
LiDAR + IMU

Orin
Camera
```

두 computer에서 data를 생성하고 ROS 2로 fusion한다면:

```text
Xavier Clock
Orin Clock
```

이 맞아야 한다.

---

# 63. Network Delay

ROS 2 message 전달에는 delay가 있다.

예:

```text
Sensor measured
     │
     ▼
Driver
     │
     ▼
DDS
     │
     ▼
Network
     │
     ▼
Subscriber
```

이 latency와 measurement timestamp를 구분해야 한다.

---

# 64. Timestamp와 Arrival Time은 다르다

예:

```text
Measurement Time = 10.000
Arrival Time     = 10.015
```

15 ms network/processing delay가 있다.

Sensor fusion에서는 보통 measurement timestamp를 사용한다.

---

# 65. Latency Compensation

실시간 system에서는 sensor latency를 측정하고
필요한 경우 estimator에서 보상할 수도 있다.

하지만 먼저:

```text
Latency가 얼마인지 측정
Timestamp source 확인
```

이 필요하다.

---

# 66. Out-of-Order Message

Network 상황에 따라 message가 예상 순서와 다르게 도착할 수도 있다.

예:

```text
Message t=10.2 arrives
then
Message t=10.1 arrives
```

Algorithm이 이런 상황을 처리할 수 있어야 할 수 있다.

---

# 67. Queue

ROS 2와 sensor driver는 message queue를 사용할 수 있다.

Processing이 느리면 오래된 message가 queue에 쌓인다.

```text
Sensor
  ↓
Queue
1
2
3
4
5
...
  ↓
Slow Processing
```

---

# 68. Queue Delay

Queue가 너무 길면 최신 sensor data가 아니라
과거 data를 처리하게 될 수 있다.

예:

```text
Current Time = 10.500

Processing Message Timestamp = 10.100
```

400 ms 뒤처진 상태다.

---

# 69. Real-Time Sensor Pipeline

좋은 pipeline은:

```text
Sensor measures
    ↓
Timestamp
    ↓
Transfer
    ↓
Processing
    ↓
Output
```

의 각 delay를 이해해야 한다.

---

# 70. End-to-End Latency

전체 delay:

```text
Sensor measurement
        ↓
Driver
        ↓
ROS 2
        ↓
Algorithm
        ↓
Control
```

까지의 시간을 end-to-end latency라고 볼 수 있다.

---

# 71. Timestamp Debugging

Sensor fusion 문제가 의심되면 다음을 확인한다.

```text
1. Sensor timestamp source?
2. Unit correct?
3. Clock domain same?
4. Timestamp monotonic?
5. Offset?
6. Drift?
7. Network delay?
8. Queue delay?
```

---

# 72. Unit 문제

Timestamp 단위가 다를 수 있다.

예:

```text
seconds
milliseconds
microseconds
nanoseconds
```

변환을 잘못하면 큰 문제가 생긴다.

예:

```text
1 second
=
1000 milliseconds
=
1,000,000 microseconds
=
1,000,000,000 nanoseconds
```

---

# 73. Millisecond

```text
ms
=
10^-3 seconds
```

예:

```text
5 ms
=
0.005 s
```

---

# 74. Microsecond

```text
us
=
10^-6 seconds
```

---

# 75. Nanosecond

```text
ns
=
10^-9 seconds
```

ROS 2 timestamp에서 nanosecond 단위를 자주 본다.

---

# 76. Timestamp Overflow

작은 integer type으로 timestamp를 저장하면
오래 실행했을 때 overflow가 날 수도 있다.

그래서 데이터 type도 중요하다.

---

# 77. Clock Reset

Sensor reboot나 driver restart 후 timestamp가 다시 0부터 시작할 수 있다.

예:

```text
100.1
100.2
100.3
0.0
0.1
```

Algorithm이 이런 reset을 처리하지 못하면 문제가 생긴다.

---

# 78. ROS Bag으로 Timing 분석

센서 topic을 rosbag으로 저장한 뒤
timestamp 간 차이를 분석할 수 있다.

예:

```text
LiDAR stamp
IMU stamp
Joint stamp
```

간의 delta를 계산한다.

---

# 79. Timing Histogram

예를 들어 LiDAR와 가장 가까운 IMU timestamp 차이를 계산한다.

```text
0.8 ms
1.2 ms
0.5 ms
4.3 ms
...
```

Histogram으로 보면 synchronization quality를 평가할 수 있다.

---

# 80. Frequency만 보면 부족하다

예:

```text
IMU = 200 Hz
```

라고 해도 timestamp가 불규칙할 수 있다.

```text
5 ms
5 ms
20 ms
1 ms
4 ms
```

평균 frequency만 맞아도 jitter가 클 수 있다.

---

# 81. Inter-Arrival Time

연속 message가 도착하는 시간 간격:

```text
t[n] - t[n-1]
```

을 확인할 수 있다.

이 값이 일정한지 보면 sensor timing stability를 알 수 있다.

---

# 82. Measurement Interval vs Arrival Interval

두 개를 구분해야 한다.

```text
Measurement Timestamp Interval
```

과:

```text
Host Arrival Interval
```

은 다를 수 있다.

Network jitter가 있으면 arrival interval이 흔들릴 수 있다.

---

# 83. FAST-LIO2 Debugging Example

Map이 이상하게 찌그러진다고 하자.

확인:

```text
LiDAR frequency normal?
IMU frequency normal?
Extrinsic correct?
Timestamp monotonic?
LiDAR-IMU offset?
Deskew active?
```

시간 문제도 주요 원인 후보다.

---

# 84. Quadruped에서 더 중요한 이유

사족보행은 몸체 motion이 부드럽지 않을 수 있다.

예:

```text
Foot impact
Body vibration
Rapid pitch/roll
Gait transition
```

짧은 시간에도 motion 변화가 크다.

그래서 timestamp error가 wheeled robot보다 더 크게 영향을 줄 수 있다.

---

# 85. Foot Impact Example

Touchdown 순간:

```text
t = 1.000
impact
```

IMU acceleration이 크게 변한다.

Contact timestamp가:

```text
1.030
```

으로 30 ms 늦다면 estimator가 잘못된 contact constraint를 사용할 수 있다.

---

# 86. LiDAR + Leg Odometry

향후 구조:

```text
LiDAR
IMU
Joint Encoder
Foot Contact
      │
      ▼
Synchronization
      │
      ▼
Leg Odometry
      │
      ▼
State Estimator
      │
      ▼
SLAM
```

이때 각 sensor clock alignment가 필수다.

---

# 87. Time Calibration

Sensor 간 일정한 time offset이 있다면
이를 calibration parameter처럼 추정할 수도 있다.

예:

```text
Camera timestamp
=
IMU timestamp + 8 ms
```

이 offset을 보정하는 방식이다.

---

# 88. Temporal Calibration

Spatial calibration:

```text
Sensor A와 B의 위치/회전 관계
```

Temporal calibration:

```text
Sensor A와 B의 시간 관계
```

이다.

둘 다 sensor fusion에 중요하다.

---

# 89. Spatial + Temporal Calibration

Sensor fusion을 정확히 하려면:

```text
Where?
+
When?
```

을 모두 알아야 한다.

즉:

```text
Extrinsic Calibration
+
Time Synchronization
```

이 필요하다.

---

# 90. Linux에서 Time 확인 명령어

현재 시간:

```bash
date
```

상태:

```bash
timedatectl
```

NTP 관련:

```bash
timedatectl status
```

환경에 따라:

```bash
chronyc tracking
```

등을 사용할 수도 있다.

---

# 91. `chrony`

Linux에서 NTP synchronization을 위해 `chrony`를 사용하는 경우가 많다.

예:

```bash
chronyc tracking
```

```bash
chronyc sources
```

로 synchronization 상태를 볼 수 있다.

---

# 92. PTP Linux Tools

PTP를 사용할 때:

```text
ptp4l
phc2sys
```

같은 tool을 볼 수 있다.

`ptp4l`:

```text
PTP network synchronization
```

`phc2sys`:

```text
PTP Hardware Clock과 system clock sync
```

에 사용될 수 있다.

---

# 93. PHC

PHC는:

**PTP Hardware Clock**

이다.

Network card가 자체 hardware clock을 제공할 수 있다.

```text
NIC
 └── PHC
```

---

# 94. PTP 전체 구조 예

```text
Grandmaster
     │
     │ PTP
     ▼
Jetson NIC PHC
     │
     ▼
phc2sys
     │
     ▼
Linux System Clock
```

---

# 95. Robot Time Architecture 문서화

실제 robot에서는 다음을 문서화하면 좋다.

| Device | Clock Source | Sync Method | Timestamp Source |
|---|---|---|---|
| Xavier | System clock | NTP/PTP | Linux time |
| Orin | System clock | NTP/PTP | Linux time |
| LiDAR | Internal/PTP | PTP | Sensor time |
| IMU | Sensor clock | driver sync | Sensor time |
| Camera | Sensor/host | trigger/PTP | Exposure time |
| MCU | MCU clock | custom sync | MCU timestamp |

---

# 96. Debugging Checklist

Sensor fusion 문제가 있으면:

```text
[ ] All clocks synchronized?
[ ] Timestamp source known?
[ ] Timestamp units correct?
[ ] No time jumps?
[ ] No clock reset?
[ ] Offset measured?
[ ] Drift measured?
[ ] Queue latency low?
[ ] Network jitter acceptable?
[ ] Hardware sync available?
```

---

# 97. Mini Practice 1

Jetson에서:

```bash
date
```

```bash
timedatectl
```

실행.

Xavier와 Orin의 시간을 비교한다.

---

# 98. Mini Practice 2

ROS 2 sensor topic:

```bash
ros2 topic echo /imu --once
```

또는 환경에 맞는 방식으로 header timestamp를 확인한다.

확인:

```text
header.stamp.sec
header.stamp.nanosec
```

---

# 99. Mini Practice 3

LiDAR와 IMU topic의 timestamp를 각각 기록하고
시간 차이를 확인한다.

예:

```text
LiDAR:
100.125

Nearest IMU:
100.123

Difference:
2 ms
```

---

# 100. Mini Practice 4

```bash
ros2 topic hz /imu
```

만 보지 말고
연속 timestamp 차이도 확인한다.

목표:

```text
200 Hz
≈
5 ms interval
```

---

# 101. 오늘의 핵심

Sensor fusion에서는:

```text
Data correctness
```

뿐 아니라:

```text
Timing correctness
```

가 중요하다.

전체 구조:

```text
Sensor
   │
Measurement
   │
Timestamp
   │
Transfer
   │
ROS 2
   │
Synchronization
   │
Estimator
```

---

# 102. 반드시 구분할 것

```text
Timestamp
≠
Arrival Time

Frequency
≠
Perfect Timing

NTP
≠
PTP

Software Timestamp
≠
Hardware Timestamp

Wall Clock
≠
Monotonic Clock

System Time
≠
Simulation Time

Spatial Calibration
≠
Temporal Calibration
```

---

# 103. Vision60 Timing Mental Model

```text
                       Vision60

 LiDAR Clock ────────┐
 IMU Clock ──────────┤
 Joint Clock ────────┤
 Contact Clock ──────┤
 Camera Clock ───────┤
                     ▼
             Time Synchronization
                     │
                     ▼
                  Jetson
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       FAST-LIO2  Leg Odom    Vision
          │          │          │
          └──────────┼──────────┘
                     ▼
               State Estimation
```

---

# 104. 지금까지 Chapter 연결

```text
Chapter 1
CPU / GPU / RAM

Chapter 2
ARM64 / x86

Chapter 3
Linux

Chapter 4
Jetson / JetPack

Chapter 5
Hardware Interfaces

Chapter 6
ROS 2

Chapter 7
CUDA / TensorRT

Chapter 8
Robot Networking

Chapter 9
Docker

Chapter 10
Debugging & Deployment

Chapter 11
Time Synchronization & Sensor Timing
```

다음 Chapter를 추가한다면 가장 자연스러운 흐름은:

```text
Chapter 12
Robot State Estimation Fundamentals
```

이다.

여기서는 지금까지의 sensor data와 timing을 실제로:

```text
IMU
LiDAR
Joint Encoder
Foot Contact
      ↓
State Estimation
      ↓
Position
Orientation
Velocity
Bias
```

로 만드는 과정을 배우면 된다.