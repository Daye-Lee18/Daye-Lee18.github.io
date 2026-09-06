---
title: "Chapter 12. Power, Thermal & Performance Management"
importance: 13
---

> **Goal:** Jetson 같은 edge computer가 단순히 "GPU가 빠른 컴퓨터"가 아니라
> 제한된 전력과 냉각 환경에서 동작하는 embedded computer라는 점을 이해한다.
>
> Power Mode, Clock, DVFS, Thermal Throttling, CPU/GPU Utilization,
> Memory Bandwidth, Bottleneck을 이해하고 실제 로봇에서
> 성능을 어떻게 측정하고 안정적으로 유지하는지 익힌다.

---

# 1. Edge Computer에서는 최고 성능만 중요하지 않다

Desktop computer에서는:

```text
성능이 부족하다
↓
더 강한 GPU
더 큰 PSU
더 큰 Cooling
```

을 사용할 수 있다.

하지만 robot에서는:

```text
Battery
Weight
Space
Temperature
Cooling
Power Supply
```

가 제한되어 있다.

따라서 edge computing에서는:

```text
Performance
+
Power
+
Temperature
+
Stability
```

를 함께 생각해야 한다.

---

# 2. Robot의 Power Budget

로봇의 battery가 공급할 수 있는 전력은 무한하지 않다.

예:

```text
Battery
  │
  ├── Motors
  ├── Jetson
  ├── LiDAR
  ├── Camera
  ├── Network
  └── Other Electronics
```

모든 장치가 같은 energy source를 공유할 수 있다.

따라서 Jetson이 사용하는 전력도 전체 robot system 관점에서 봐야 한다.

---

# 3. Power와 Energy는 다르다

Power:

```text
Watt
W
```

Energy:

```text
Watt-hour
Wh
```

이다.

Power는:

> 지금 얼마나 빠르게 energy를 사용하고 있는가?

Energy는:

> 총 얼마만큼의 energy가 있는가?

라고 생각하면 된다.

---

# 4. 간단한 예

Battery:

```text
500 Wh
```

Robot 전체 소비 전력:

```text
250 W
```

라고 단순하게 가정하면:

```text
500 Wh / 250 W
≈ 2 h
```

이다.

실제로는 battery efficiency, voltage variation, motor load 등 때문에
이렇게 정확히 나오지는 않는다.

하지만 power budget을 이해하는 기본적인 계산이다.

---

# 5. Watt

전력의 기본 단위:

```text
W = Watt
```

전기적으로는:

```text
Power = Voltage × Current
```

즉:

```text
P = V × I
```

이다.

예:

```text
20 V × 3 A
=
60 W
```

---

# 6. Jetson도 Power Limit이 있다

Jetson은 무조건 최대 clock으로 계속 동작하지 않는다.

시스템은 상황에 따라:

```text
CPU Clock
GPU Clock
Memory Clock
Power Consumption
```

을 조절할 수 있다.

이유:

```text
Power 제한
Temperature 제한
Efficiency
Battery
```

때문이다.

---

# 7. Clock Frequency

CPU/GPU에는 clock frequency가 있다.

예:

```text
CPU
2.0 GHz

GPU
1.2 GHz
```

일반적으로 clock이 높아지면 더 많은 연산을 빠르게 수행할 가능성이 있다.

하지만:

```text
Clock ↑
→ Power ↑
→ Heat ↑
```

경향도 있다.

---

# 8. Clock이 2배면 성능도 2배인가?

아니다.

성능은:

```text
CPU
GPU
Memory
Algorithm
Parallelism
Cache
Bandwidth
I/O
```

등 여러 요소의 영향을 받는다.

따라서:

```text
Clock × 2
≠
Performance × 2
```

이다.

---

# 9. Dynamic Clock

현대 CPU/GPU는 workload에 따라 clock을 동적으로 변경할 수 있다.

예:

```text
Idle
CPU 500 MHz

Heavy Load
CPU 2 GHz
```

처럼 동작할 수 있다.

---

# 10. DVFS

DVFS:

```text
Dynamic Voltage and Frequency Scaling
```

이다.

필요한 performance에 따라:

```text
Voltage
+
Frequency
```

를 동적으로 조절한다.

---

# 11. 왜 Voltage까지 바꿀까?

일반적으로 frequency를 높이려면 더 높은 voltage가 필요할 수 있다.

그리고 power consumption은 voltage에 매우 민감하다.

개념적으로:

```text
Higher Frequency
      ↓
Higher Voltage may be required
      ↓
More Power
      ↓
More Heat
```

라고 이해하면 된다.

---

# 12. Idle 상태

Robot이 서 있고 perception workload도 거의 없다면:

```text
CPU Load ↓
GPU Load ↓
```

시스템은 clock을 낮출 수 있다.

목적:

```text
Power 절약
Heat 감소
```

---

# 13. Heavy Load 상태

예:

```text
Camera AI
+
LiDAR Processing
+
SLAM
+
Navigation
```

이 동시에 실행되면:

```text
CPU Load ↑
GPU Load ↑
Memory Traffic ↑
```

시스템이 더 높은 performance state를 사용할 수 있다.

---

# 14. Jetson Power Mode

Jetson에는 hardware와 JetPack 버전에 따라 여러 power mode가 제공될 수 있다.

예를 들어 개념적으로:

```text
Low Power Mode
Balanced Mode
Higher Performance Mode
```

처럼 CPU/GPU 자원과 power budget을 다르게 설정할 수 있다.

정확한 mode 이름과 power limit은 Jetson 모델 및 JetPack 버전에 따라 다르므로
실제 장치에서 확인해야 한다.

---

# 15. `nvpmodel`

Jetson에서 power mode를 관리할 때 자주 사용하는 도구:

```bash
nvpmodel
```

현재 mode 확인:

```bash
sudo nvpmodel -q
```

---

# 16. Power Mode가 바꾸는 것

Power mode에 따라 시스템이 사용할 수 있는:

```text
CPU Core
CPU Frequency
GPU Frequency
Memory-related limits
Power Budget
```

등이 달라질 수 있다.

정확한 설정은 Jetson 모델마다 다르다.

---

# 17. 왜 Benchmark에서 Power Mode를 기록해야 할까?

FAST-LIO2를 테스트했다고 하자.

Experiment A:

```text
High-performance power mode
```

Experiment B:

```text
Low-power mode
```

라면 실행 시간 차이가 algorithm 때문인지
power configuration 때문인지 구분하기 어렵다.

따라서 benchmark에는:

```text
Jetson Model
JetPack Version
Power Mode
Clock Configuration
Temperature
```

도 기록하는 것이 좋다.

---

# 18. `jetson_clocks`

Jetson에서는 다음 명령을 볼 수 있다.

```bash
sudo jetson_clocks
```

지원되는 환경에서는 CPU/GPU/메모리 관련 clock을
현재 power mode에서 허용되는 높은 값으로 고정하는 데 사용할 수 있다.

Benchmark를 안정적으로 비교할 때 유용할 수 있다.

---

# 19. `jetson_clocks` = 무조건 최고 성능?

정확히는 그렇게 단순하게 보면 안 된다.

```text
nvpmodel
→ Power / performance envelope 설정

jetson_clocks
→ 해당 configuration 안에서 clock scaling을 제한/고정
```

으로 이해하는 것이 좋다.

또 thermal/power limit은 여전히 존재할 수 있다.

---

# 20. Benchmark와 실제 Deployment는 다르다

Benchmark에서는:

```text
Maximum / Fixed Clock
```

이 유용할 수 있다.

하지만 실제 robot에서는:

```text
Battery Runtime
Temperature
Noise
Reliability
```

도 중요하다.

따라서 항상 최대 clock으로 운영하는 것이 최선은 아니다.

---

# 21. Heat는 어디서 생길까?

CPU와 GPU가 연산하면 electrical energy 일부가 heat로 변한다.

```text
Computation
    ↓
Power Consumption
    ↓
Heat
```

load가 높아질수록 temperature가 상승할 수 있다.

---

# 22. Cooling

발생한 heat를 밖으로 빼내야 한다.

방법:

```text
Heat Sink
Fan
Thermal Interface Material
Air Flow
Chassis
```

등이 있다.

---

# 23. Passive Cooling

Fan 없이 heat sink 등을 이용해 냉각:

```text
Chip
 ↓
Heat Sink
 ↓
Air
```

장점:

```text
조용함
Moving part 적음
```

단점:

```text
높은 sustained load에서 냉각 한계
```

가 있을 수 있다.

---

# 24. Active Cooling

Fan 등을 사용한다.

```text
Chip
 ↓
Heat Sink
 ↓
Fan
 ↓
Air Flow
```

높은 sustained performance를 유지하는 데 도움이 된다.

---

# 25. Robot에서는 Cooling이 더 어렵다

Desktop:

```text
Large Case
Large Fans
Stable Room
```

Robot:

```text
Small enclosure
Dust
Outdoor heat
Direct sunlight
Limited airflow
Vibration
```

환경일 수 있다.

따라서 실제 현장에서는 thermal management가 더 중요하다.

---

# 26. Temperature

Jetson에는 여러 thermal sensor가 있을 수 있다.

예:

```text
CPU Temperature
GPU Temperature
SoC Temperature
Board Temperature
```

정확한 sensor 종류는 모델마다 다를 수 있다.

---

# 27. Thermal Throttling

Temperature가 너무 높아지면 hardware 보호를 위해
clock을 낮출 수 있다.

이를:

```text
Thermal Throttling
```

이라고 한다.

구조:

```text
Heavy Load
   ↓
Temperature ↑
   ↓
Thermal Limit
   ↓
Clock ↓
   ↓
Performance ↓
```

---

# 28. 왜 Thermal Throttling이 위험할까?

예를 들어 처음 FAST-LIO2 실행:

```text
100 Hz
```

10분 후:

```text
65 Hz
```

로 떨어진다고 하자.

Code는 바뀌지 않았다.

원인은:

```text
Temperature ↑
↓
Clock ↓
↓
Processing Rate ↓
```

일 수 있다.

---

# 29. Sustained Performance

짧은 benchmark에서 빠른 것과
30분 동안 같은 성능을 유지하는 것은 다르다.

```text
Peak Performance
≠
Sustained Performance
```

Robot에서는 sustained performance가 매우 중요하다.

---

# 30. `tegrastats`

Jetson monitoring에서 매우 중요한 도구:

```bash
tegrastats
```

실행:

```bash
tegrastats
```

환경에 따라 다음과 같은 정보를 볼 수 있다.

```text
RAM
CPU
GPU
Temperature
Power-related metrics
Memory controller activity
```

표현 형식은 Jetson 모델과 JetPack 버전에 따라 다를 수 있다.

---

# 31. CPU Utilization

예:

```text
CPU0 90%
CPU1 20%
CPU2 10%
CPU3 5%
```

이런 결과라면 전체 CPU가 부족하다기보다
특정 thread가 하나의 core를 강하게 사용하고 있을 수 있다.

---

# 32. Single-Thread Bottleneck

CPU core가 8개여도 application의 핵심 부분이 single-thread라면:

```text
Core 0 = 100%

Other cores = mostly idle
```

일 수 있다.

이 경우 CPU 전체 평균만 보면 문제를 놓칠 수 있다.

---

# 33. Multi-Thread

작업을 여러 thread로 나누면 여러 CPU core를 활용할 수 있다.

```text
Task
 ├── Thread 1 → Core 0
 ├── Thread 2 → Core 1
 ├── Thread 3 → Core 2
 └── Thread 4 → Core 3
```

하지만 모든 algorithm을 완벽하게 병렬화할 수 있는 것은 아니다.

---

# 34. GPU Utilization

GPU workload가 있다면 GPU utilization을 확인한다.

예:

```text
Camera
 ↓
Neural Network
 ↓
GPU
```

GPU utilization이 높다고 무조건 나쁜 것은 아니다.

GPU를 사용하기 위해 만든 workload라면 높은 utilization은 정상일 수 있다.

---

# 35. Utilization 100%의 의미

```text
GPU = 100%
```

이면 GPU가 바쁘다는 뜻이다.

하지만:

```text
100% = 문제
```

라는 뜻은 아니다.

중요한 것은:

```text
Deadline을 만족하는가?
Latency가 acceptable한가?
Temperature가 안정적인가?
```

이다.

---

# 36. CPU 100%도 같은 원리

CPU core 하나가:

```text
100%
```

라고 해서 반드시 문제가 아니다.

문제는:

```text
Processing deadline miss
Queue 증가
Sensor data drop
Latency 증가
```

가 발생하는가이다.

---

# 37. Bottleneck

System performance를 가장 크게 제한하는 부분을:

```text
Bottleneck
```

이라고 한다.

예:

```text
Fast CPU
Fast GPU
Slow Network
```

이면 network가 bottleneck일 수 있다.

---

# 38. Robot Pipeline Bottleneck

예:

```text
Camera
30 FPS
  ↓
Preprocessing
30 FPS
  ↓
Inference
12 FPS
  ↓
Postprocessing
12 FPS
```

이 경우 inference가 bottleneck일 가능성이 높다.

---

# 39. FAST-LIO2 Pipeline

예:

```text
LiDAR
10 Hz
  ↓
Point Cloud Preprocessing
  ↓
IMU Synchronization
  ↓
State Estimation
  ↓
Map Update
  ↓
Odometry
```

어느 단계가 가장 많은 CPU time을 사용하는지 확인해야 한다.

---

# 40. Profiling

어디서 시간이 소비되는지 측정하는 것을:

```text
Profiling
```

이라고 한다.

Optimization 전에 profiling을 해야 한다.

---

# 41. 가장 중요한 Optimization 원칙

```text
Measure First
Optimize Second
```

이다.

감으로:

```text
"GPU로 옮기면 빨라질 것 같다."
```

라고 결정하지 않는다.

먼저:

```text
어디가 느린가?
```

를 측정한다.

---

# 42. Latency

하나의 작업을 완료하는 데 걸리는 시간:

```text
Latency
```

예:

```text
One camera frame inference
=
25 ms
```

---

# 43. Throughput

일정 시간 동안 처리할 수 있는 작업량:

```text
Throughput
```

예:

```text
40 frames/s
```

---

# 44. Latency와 Throughput은 다르다

예:

```text
Batch 1
Latency = 10 ms
Throughput = 100 items/s
```

Batching을 하면:

```text
Latency ↑
Throughput ↑
```

가 될 수도 있다.

Robot에서는 latency가 특히 중요할 수 있다.

---

# 45. FPS

FPS:

```text
Frames Per Second
```

예:

```text
30 FPS
```

이면 초당 30 frame을 처리한다.

한 frame당 평균 시간은 단순 계산하면:

```text
1 / 30 s
≈ 33.3 ms
```

이다.

---

# 46. Deadline

Real-time pipeline에서는 특정 시간 안에 결과가 필요할 수 있다.

예:

```text
Camera = 30 FPS

Frame interval
≈ 33.3 ms
```

처리가:

```text
50 ms
```

걸리면 새 frame이 처리 속도보다 빠르게 들어온다.

---

# 47. Queue가 쌓이는 상황

```text
Input
30 FPS

Processing
20 FPS
```

이면:

```text
Input
↓↓↓↓↓↓↓↓

Queue
████████

Processing
↓↓↓↓
```

queue가 계속 증가할 수 있다.

---

# 48. 결과

Queue가 증가하면:

```text
Latency ↑
Memory Usage ↑
Old Data Processing
```

문제가 생긴다.

로봇에서는 오래된 sensor data를 처리하는 것이 특히 위험할 수 있다.

---

# 49. Drop 전략

실시간 perception에서는 오래된 frame을 모두 처리하는 것보다
일부 frame을 버리고 최신 데이터를 처리하는 것이 나을 수도 있다.

```text
Old
X X X

Latest
✓
```

application requirement에 따라 결정한다.

---

# 50. Memory Bandwidth

CPU와 GPU가 아무리 빠르더라도
memory에서 데이터를 충분히 빠르게 가져오지 못하면 성능이 제한된다.

이를 이해하기 위해:

```text
Memory Bandwidth
```

가 중요하다.

---

# 51. Bandwidth

Memory bandwidth는 대략:

> 단위 시간 동안 memory와 processor 사이에서 얼마나 많은 데이터를 이동할 수 있는가?

를 나타낸다.

예:

```text
GB/s
```

---

# 52. Point Cloud는 Memory Traffic이 크다

LiDAR point cloud:

```text
Millions of points
```

를 반복해서:

```text
Read
Transform
Search
Copy
Write
```

하면 memory traffic이 커진다.

---

# 53. Camera도 마찬가지다

고해상도 image:

```text
1920 × 1080
```

을 여러 pipeline에서 복사하면:

```text
Camera
 ↓ copy
CPU
 ↓ copy
GPU
 ↓ copy
Output
```

memory bandwidth와 latency를 많이 사용할 수 있다.

---

# 54. Zero-Copy가 등장하는 이유

불필요한 memory copy를 줄이기 위해:

```text
Zero-Copy
```

같은 기술을 고려할 수 있다.

목표:

```text
Copy
Copy
Copy
```

를 줄이는 것이다.

---

# 55. 하지만 Shared Memory = Zero-Copy는 아니다

Jetson은 CPU와 GPU가 physical memory를 공유하는 architecture를 사용하지만:

```text
Shared Physical Memory
≠
Automatic Zero-Copy
```

이다.

Software framework와 memory allocation 방식에 따라 실제 copy가 발생할 수 있다.

---

# 56. CPU Bottleneck

예:

```text
CPU 100%
GPU 20%
```

이고 application이 느리다면 CPU가 bottleneck일 가능성이 있다.

하지만 utilization만으로 확정하지 말고 profiling해야 한다.

---

# 57. GPU Bottleneck

예:

```text
CPU 30%
GPU 100%
Inference latency high
```

이면 GPU workload가 bottleneck 후보가 된다.

---

# 58. Memory Bottleneck

예:

```text
CPU arithmetic utilization 낮음
GPU arithmetic utilization 낮음

하지만
Memory controller activity 높음
```

이면 memory bandwidth가 bottleneck일 수 있다.

---

# 59. I/O Bottleneck

예:

```text
rosbag record
+
Camera
+
LiDAR
```

중 storage write가 느리면:

```text
Disk I/O
```

가 bottleneck이 될 수 있다.

이 내용은 Chapter 13에서 더 자세히 다룬다.

---

# 60. Network Bottleneck

예:

```text
Camera 500 Mbps
LiDAR 200 Mbps
Other ROS Traffic 400 Mbps
```

인데 1 Gbps network를 사용한다고 하자.

Protocol overhead와 실제 link utilization까지 고려하면
network가 bottleneck이 될 수 있다.

---

# 61. End-to-End Performance

Robot에서는 component 하나만 빠르면 충분하지 않다.

```text
Sensor
  ↓
Driver
  ↓
ROS 2
  ↓
Preprocessing
  ↓
AI / SLAM
  ↓
Planning
  ↓
Control
```

전체 latency를 봐야 한다.

---

# 62. Peak Performance

아주 짧은 시간 동안 낼 수 있는 최대 성능:

```text
Peak Performance
```

---

# 63. Sustained Performance

오랫동안 안정적으로 유지 가능한 성능:

```text
Sustained Performance
```

Robot에서는:

```text
Peak
```

보다:

```text
Sustained
```

가 더 중요한 경우가 많다.

---

# 64. Performance per Watt

Edge computing에서는:

```text
Performance / Watt
```

도 중요하다.

같은 workload를:

```text
Device A
50 W

Device B
20 W
```

로 처리할 수 있다면,
성능이 비슷한 경우 B가 battery robot에 더 유리할 수 있다.

---

# 65. Efficiency

따라서 edge AI에서는:

```text
Maximum Performance
```

만 보는 것이 아니라:

```text
Performance per Watt
```

를 중요하게 본다.

---

# 66. CPU vs GPU 선택

예:

```text
ROS callbacks
State machine
Serial communication
```

→ CPU가 적합.

```text
Neural Network
Image processing
Large parallel matrix operations
```

→ GPU가 적합할 가능성이 높다.

---

# 67. GPU로 옮긴다고 항상 빨라지지 않는다

GPU 실행에는:

```text
Kernel Launch
Synchronization
Memory Access
Data Transfer
```

overhead가 있다.

작은 task라면 CPU가 더 빠를 수도 있다.

---

# 68. FAST-LIO2와 GPU

FAST-LIO2를 Jetson에서 실행한다고 해서:

```text
GPU가 자동 사용
```

되는 것은 아니다.

기본 implementation의 핵심 연산이 CPU code라면
GPU utilization이 거의 없을 수도 있다.

---

# 69. GPU가 놀고 있어도 문제는 아니다

예:

```text
FAST-LIO2

CPU 300%
GPU 0%
```

라고 해서 GPU configuration이 잘못된 것은 아니다.

Algorithm이 GPU를 사용하도록 구현되지 않았을 수 있다.

---

# 70. CPU 300%?

Linux monitoring tool에서는 multi-core CPU 사용률을 합산해서 표현할 수 있다.

예:

```text
100%
≈ 한 core를 완전히 사용

300%
≈ 약 세 core 상당의 CPU time
```

단, 정확한 표시 방식은 tool마다 확인해야 한다.

---

# 71. Thread Scheduling

Linux scheduler는 thread를 CPU core에 배치한다.

```text
Thread A → CPU 0
Thread B → CPU 3
Thread C → CPU 5
```

상황에 따라 core 사이를 이동할 수도 있다.

---

# 72. Context Switch

CPU가 실행 중인 task를 바꾸는 것을:

```text
Context Switch
```

라고 한다.

너무 많은 thread/process가 경쟁하면 context switching overhead가 증가할 수 있다.

---

# 73. More Threads ≠ Always Faster

Thread를 많이 만든다고 무조건 빨라지는 것은 아니다.

```text
More Threads
→ Parallelism ↑
```

가능성이 있지만:

```text
Synchronization
Lock contention
Context switching
Cache misses
```

도 증가할 수 있다.

---

# 74. Thermal Test

실제 robot에서는 짧은 1분 benchmark보다:

```text
30 min
1 hour
```

같은 sustained test가 유용하다.

기록:

```text
Time
CPU Load
GPU Load
Temperature
Clock
Power
Latency
```

---

# 75. Thermal Test 예

```text
0 min
GPU 70%
55°C
Inference 20 ms

10 min
GPU 70%
68°C
Inference 20 ms

30 min
GPU 70%
78°C
Inference 23 ms

45 min
GPU 70%
Thermal limit
Inference 35 ms
```

이런 변화가 있다면 thermal 문제가 의심된다.

---

# 76. Temperature만 보면 부족하다

다음도 같이 봐야 한다.

```text
Temperature
Clock
Utilization
Latency
Power
```

예:

```text
Temperature ↑
Clock ↓
Latency ↑
```

이면 throttling을 강하게 의심할 수 있다.

---

# 77. Ambient Temperature

같은 Jetson이라도:

```text
Office 22°C
```

와:

```text
Construction Site 38°C
```

에서는 냉각 성능이 다르다.

Benchmark 환경의 ambient temperature도 중요하다.

---

# 78. Enclosure

Jetson을 밀폐된 enclosure에 넣으면:

```text
Dust protection ↑
```

에는 좋을 수 있지만:

```text
Airflow ↓
Temperature ↑
```

문제가 생길 수 있다.

Mechanical design과 computing design이 연결되는 부분이다.

---

# 79. Dust

건설 현장에서는 dust가:

```text
Fan
Heat Sink
Air Intake
```

에 쌓일 수 있다.

시간이 지나면 cooling performance가 저하될 수 있다.

---

# 80. Power Supply 안정성

Jetson이 충분한 power를 받지 못하면:

```text
Unexpected shutdown
Instability
Performance limitation
```

문제가 생길 수 있다.

따라서 단순히 battery capacity만 볼 것이 아니라:

```text
Voltage
Current capability
Peak load
Regulator
Cable
Connector
```

도 중요하다.

---

# 81. 순간적인 Peak Load

평균 전력이:

```text
30 W
```

여도 순간적으로 더 높은 current가 필요할 수 있다.

Power supply가 이를 감당하지 못하면 system instability가 발생할 수 있다.

---

# 82. Brownout

공급 voltage가 순간적으로 너무 떨어지는 현상을:

```text
Brownout
```

이라고 한다.

Robot motor가 큰 current를 순간적으로 사용하면
power architecture가 좋지 않은 경우 compute 쪽에도 영향을 줄 수 있다.

---

# 83. Motor와 Compute Power

개념적으로:

```text
Battery
   │
   ├── Motor Power
   │
   └── DC/DC
        │
        ▼
      Jetson
```

처럼 compute power를 안정적으로 공급하도록 설계할 수 있다.

정확한 Vision60 내부 power architecture는 실제 hardware documentation을 확인해야 한다.

---

# 84. Performance Monitoring Stack

Jetson에서:

```text
Application
    │
    ▼
ROS 2 / CUDA
    │
    ▼
CPU / GPU / RAM
    │
    ▼
Clock / Power
    │
    ▼
Temperature
```

를 함께 모니터링한다.

---

# 85. Linux Tools

CPU/process:

```bash
htop
```

Memory:

```bash
free -h
```

Jetson:

```bash
tegrastats
```

Power mode:

```bash
sudo nvpmodel -q
```

Disk:

```bash
df -h
```

Network:

```bash
ip -s link
```

등을 조합할 수 있다.

---

# 86. `tegrastats`를 볼 때 질문

단순히 숫자를 읽는 것이 아니라 질문한다.

```text
CPU가 꽉 찼나?

GPU가 실제로 사용되나?

RAM이 부족한가?

Temperature가 계속 상승하나?

Clock이 내려가나?

Memory controller가 bottleneck인가?
```

---

# 87. Application Metric도 같이 봐야 한다

System metric:

```text
CPU
GPU
RAM
Temperature
```

만으로는 부족하다.

Application metric:

```text
LiDAR Hz
Odometry Hz
Inference FPS
Planning latency
Dropped frames
```

도 함께 기록한다.

---

# 88. 예: FAST-LIO2 Profiling

기록:

```text
LiDAR Input
10 Hz

IMU
200 Hz

Odometry Output
100 Hz

CPU
220%

GPU
0%

RAM
3.5 GB

Temperature
62°C
```

이런 식으로 baseline을 만든다.

---

# 89. AI + SLAM 동시 실행

예:

```text
FAST-LIO2
→ CPU

Object Detection
→ GPU

Camera Preprocessing
→ CPU/GPU

Nav2
→ CPU
```

동시에 실행하면 서로 resource를 공유한다.

---

# 90. Resource Contention

여러 application이 같은 resource를 경쟁하는 것을:

```text
Resource Contention
```

이라고 한다.

예:

```text
FAST-LIO2
      \
       CPU
      /
Nav2
```

또는:

```text
Camera AI
       \
        Memory Bandwidth
       /
Point Cloud Processing
```

---

# 91. 단독 Benchmark만 보면 안 되는 이유

FAST-LIO2만 실행:

```text
100 Hz
```

AI까지 실행:

```text
70 Hz
```

가 될 수 있다.

실제 robot workload 조합으로도 benchmark해야 한다.

---

# 92. Worst-Case Scenario

Robot system에서는 평균적인 상황뿐 아니라:

```text
High-resolution camera
Dense point cloud
Large map
Multiple ROS nodes
Recording rosbag
Hot environment
```

이 동시에 발생하는 worst-case도 테스트해야 한다.

---

# 93. Headroom

System을 항상 100% resource로 운영하면
갑작스러운 workload 증가에 대응하기 어렵다.

따라서 일정한:

```text
Performance Headroom
```

을 남기는 것이 좋다.

---

# 94. Headroom 예

평상시:

```text
CPU 60%
GPU 65%
RAM 55%
```

정도로 운영하고,
순간적인 workload 증가를 처리할 여유를 남기는 식이다.

정확한 적정 비율은 application 요구사항에 따라 다르다.

---

# 95. Performance Regression

Software update 이후:

```text
Before
FAST-LIO2 CPU 180%

After
FAST-LIO2 CPU 280%
```

가 되었다면:

```text
Performance Regression
```

일 수 있다.

기능이 정상이어도 성능이 악화된 것이다.

---

# 96. Regression Test

Software version마다 동일한 benchmark를 실행한다.

```text
Same Dataset
Same Config
Same Power Mode
Same Hardware
Same Environment
```

조건을 최대한 맞춘다.

---

# 97. Benchmark 기록 예

| Item | Value |
|---|---|
| Device | Jetson AGX Orin |
| Software Commit | abc123 |
| Power Mode | Recorded mode |
| Input | Same rosbag |
| LiDAR Rate | 10 Hz |
| IMU Rate | 200 Hz |
| CPU Usage | measured |
| GPU Usage | measured |
| RAM | measured |
| Max Temperature | measured |
| Output Rate | measured |
| P95 Latency | measured |

이런 표를 software version별로 비교할 수 있다.

---

# 98. P95 Latency

100번 측정했다고 하자.

```text
대부분 10 ms

일부 30 ms
```

평균만 보면 spike를 놓칠 수 있다.

P95는 대략:

> 측정값의 95%가 이 값 이하에 들어오는 latency

를 의미한다.

---

# 99. 왜 Robot에서 Tail Latency가 중요한가?

평균:

```text
10 ms
```

라도 가끔:

```text
300 ms
```

가 발생하면 control/perception pipeline에 문제가 될 수 있다.

그래서:

```text
Average
P95
P99
Maximum
```

을 함께 보는 것이 좋다.

---

# 100. Optimization 순서

성능 문제가 발생하면:

```text
1. Measure
      ↓
2. Find Bottleneck
      ↓
3. Form Hypothesis
      ↓
4. Optimize
      ↓
5. Measure Again
```

순서로 진행한다.

---

# 101. 나쁜 Optimization

```text
FAST-LIO2가 느리다.
↓
GPU로 옮긴다.
```

이렇게 바로 결정하면 안 된다.

실제로 bottleneck이:

```text
Network
Disk
Lock
Memory allocation
Single-threaded map search
```

일 수도 있다.

---

# 102. 좋은 Optimization

예:

```text
Profiling
↓
Nearest-neighbor search = 45% CPU time
↓
Algorithm/data structure 개선 검토
↓
Benchmark
↓
Latency 20% 감소
```

처럼 근거를 가지고 진행한다.

---

# 103. Vision60 Example

개념적으로:

```text
                     Vision60

                       Battery
                          │
          ┌───────────────┴──────────────┐
          │                              │
          ▼                              ▼
      Locomotion                       Compute
                                         │
                                         ▼
                                      Jetson
                                         │
                      ┌──────────────────┼──────────────────┐
                      │                  │                  │
                      ▼                  ▼                  ▼
                   FAST-LIO2          Nav2              Vision AI
                      │                  │                  │
                      └──────────────────┼──────────────────┘
                                         │
                                         ▼
                               CPU / GPU / Memory
                                         │
                                         ▼
                                Power Consumption
                                         │
                                         ▼
                                      Heat
                                         │
                                         ▼
                                     Cooling
```

---

# 104. Performance Debugging Mental Model

Robot이 느려졌다면:

```text
Application workload increased?
          ↓
CPU bottleneck?
          ↓
GPU bottleneck?
          ↓
Memory bottleneck?
          ↓
Network bottleneck?
          ↓
Disk bottleneck?
          ↓
Power mode changed?
          ↓
Temperature high?
          ↓
Thermal throttling?
```

순서로 확인할 수 있다.

---

# 105. Mini Practice 1

Jetson에서:

```bash
sudo nvpmodel -q
```

를 실행한다.

확인:

```text
현재 Power Mode는 무엇인가?
```

---

# 106. Mini Practice 2

```bash
tegrastats
```

를 실행한다.

FAST-LIO2나 다른 workload를 실행하면서:

```text
CPU
GPU
RAM
Temperature
```

변화를 관찰한다.

---

# 107. Mini Practice 3

다음 세 상태를 비교한다.

```text
1. Idle

2. FAST-LIO2 only

3. FAST-LIO2 + AI
```

기록:

```text
CPU
GPU
RAM
Temperature
```

---

# 108. Mini Practice 4

FAST-LIO2를 30분 이상 실행하면서:

```text
Time
Temperature
Odometry Hz
CPU Usage
```

를 기록한다.

질문:

```text
시간이 지나면서 성능이 떨어지는가?

Temperature와 관계가 있는가?
```

---

# 109. Mini Practice 5

같은 workload를:

```text
Power Mode A

vs

Power Mode B
```

에서 실행해본다.

비교:

```text
Latency
Power
Temperature
```

단, 실제 로봇에서 power mode를 변경할 때는
hardware 요구사항과 운영 정책을 먼저 확인한다.

---

# 110. 반드시 구분할 것

```text
Power
≠
Energy

Clock
≠
Performance

Utilization
≠
Performance

Peak Performance
≠
Sustained Performance

Latency
≠
Throughput

CPU 100%
≠
System Failure

GPU 0%
≠
GPU Problem

Shared Memory
≠
Automatic Zero-Copy

High Temperature
≠
Thermal Throttling

Benchmark
≠
Real Robot Workload
```

---

# 111. 오늘의 핵심

Edge computer의 performance는 단순히:

```text
CPU가 빠른가?
GPU가 빠른가?
```

만으로 결정되지 않는다.

실제로는:

```text
                 Application
                      │
                      ▼
                CPU / GPU
                      │
                      ▼
             Memory / I/O
                      │
                      ▼
                 Clock
                      │
                      ▼
                 Power
                      │
                      ▼
                  Heat
                      │
                      ▼
                 Cooling
                      │
                      ▼
          Sustained Performance
```

로 연결되어 있다.

---

# 112. Edge Computing의 핵심 관점

Cloud server에서는:

```text
더 많은 compute resource
```

를 추가하기 상대적으로 쉽다.

하지만 robot edge computer에서는:

```text
Limited Power
Limited Cooling
Limited Space
Limited Weight
Limited Network
```

안에서 필요한 performance를 만들어야 한다.

그래서:

```text
Performance per Watt
+
Sustained Performance
```

가 중요하다.

---

# 113. Chapter 12 Mental Model

최종적으로 Jetson을 볼 때:

```text
Workload
   │
   ▼
CPU / GPU / Memory
   │
   ▼
Power Consumption
   │
   ▼
Heat Generation
   │
   ▼
Cooling
   │
   ▼
Temperature
   │
   ▼
Clock / Throttling
   │
   ▼
Application Performance
```

이 loop를 생각한다.

---

# Next Chapter

## Chapter 13. Storage & Data Logging

다음 Chapter에서는 robot이 생성하는 대용량 데이터를 다룬다.

```text
LiDAR Point Cloud
Camera Video
ROS Bag
Map
Logs
AI Dataset
```

그리고:

```text
SSD
NVMe
eMMC
Filesystem
Read / Write Speed
IOPS
Write Endurance
Mount
Partition
Disk Full
Log Rotation
rosbag2
Data Retention
```

을 연결한다.

특히:

```text
"LiDAR + Camera를 계속 저장하면
왜 Jetson storage가 금방 차는가?"
```

와:

```text
"NVMe가 빠르다는 것은 정확히 무엇이 빠른 것인가?"
```

를 이해하는 것이 Chapter 13의 목표다.
