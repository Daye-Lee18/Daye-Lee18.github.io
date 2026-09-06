---
title: "Chapter 15. Real-Time Computing"
impotance: 16
---

> **Goal:** 로봇에서 "빠르다"와 "정해진 시간 안에 반드시 실행된다"의 차이를 이해한다.
>
> Real-Time, Latency, Jitter, Deadline, Scheduler, Priority, Context Switch,
> Interrupt Latency, CPU Affinity, Priority Inversion, PREEMPT_RT, Watchdog 개념을 이해하고,
> 왜 Jetson과 MCU의 역할을 분리하는지 연결해서 이해하는 것이 목표다.

---

# 1. Real-Time은 단순히 빠르다는 뜻이 아니다

많이 하는 오해:

```text
Real-Time
=
Very Fast
```

가 아니다.

Real-Time system에서 가장 중요한 것은:

> 계산 결과가 정해진 시간 안에 나오는가?

이다.

즉 평균 속도보다:

```text
Deadline
Determinism
Worst-case latency
```

가 중요하다.

---

# 2. Fast vs Real-Time

예를 들어 프로그램 A:

```text
Average execution time = 1 ms
```

하지만 가끔:

```text
100 ms
```

까지 걸린다고 하자.

프로그램 B:

```text
Always between 8~9 ms
```

라고 하자.

10 ms 안에 반드시 결과가 필요하다면:

```text
Program A
→ 평균적으로 훨씬 빠름
→ 하지만 deadline miss 가능

Program B
→ 느리지만 predictable
→ deadline 만족
```

이다.

Real-Time에서는 B가 더 적합할 수 있다.

---

# 3. Deadline

Deadline은:

> 작업이 반드시 완료되어야 하는 시간 제한

이다.

예:

```text
Control loop = 100 Hz
```

이면 period는:

```text
1 / 100
=
0.01 s
=
10 ms
```

이다.

즉 각 control cycle이 약 10 ms 안에 끝나야 다음 cycle을 제때 처리할 수 있다.

---

# 4. 100 Hz의 의미

```text
100 Hz
```

는:

```text
초당 100번
```

이라는 뜻이다.

Period:

```text
T = 1 / f
```

따라서:

```text
100 Hz
→ 10 ms

200 Hz
→ 5 ms

50 Hz
→ 20 ms

10 Hz
→ 100 ms
```

이다.

---

# 5. "100 Hz로 동작한다"와 "매 10 ms마다 실행된다"는 다르다

예를 들어 1초 동안 100번 실행되더라도 timing이:

```text
1 ms
1 ms
1 ms
50 ms
1 ms
...
```

처럼 불규칙할 수 있다.

평균 frequency는 100 Hz 근처일 수 있지만
real-time requirement는 만족하지 못할 수 있다.

---

# 6. Latency

Latency는:

> 어떤 event가 발생한 뒤 원하는 response가 나오기까지 걸리는 시간

이다.

예:

```text
Sensor measurement
      │
      ▼
Processing
      │
      ▼
Control command
```

전체가:

```text
8 ms
```

걸렸다면 end-to-end latency는 약 8 ms다.

---

# 7. Jitter

같은 작업의 latency가 매번 달라지는 정도를:

```text
Jitter
```

라고 한다.

예:

```text
10 ms
11 ms
9 ms
10 ms
45 ms
```

평균은 괜찮아도 45 ms spike가 있으면
real-time control에서는 문제가 될 수 있다.

---

# 8. Determinism

Determinism은:

> 시스템 timing이 얼마나 예측 가능한가?

와 관련된다.

예:

```text
Execution time:
9.8
10.1
10.0
9.9
```

는 predictable하다.

반면:

```text
3
2
80
4
2
```

는 average는 빠를 수 있지만 predictable하지 않다.

---

# 9. Hard Real-Time

Hard Real-Time에서는:

> Deadline miss가 절대 허용되지 않거나 매우 심각하다.

예:

```text
Motor current control
Safety-critical actuation
Industrial motion control
```

deadline을 놓치면:

```text
Instability
Equipment damage
Safety issue
```

가 발생할 수 있다.

---

# 10. Soft Real-Time

Soft Real-Time에서는 deadline miss가 좋지 않지만
가끔 발생해도 시스템이 완전히 실패하지는 않는다.

예:

```text
Camera perception
Video streaming
SLAM visualization
```

몇 frame이 늦어질 수는 있지만
즉시 시스템 전체 failure로 이어지지는 않을 수 있다.

---

# 11. Firm Real-Time

Firm Real-Time이라는 표현도 있다.

Deadline을 지나면 결과의 가치가 거의 없어지는 시스템이다.

예:

```text
오래된 obstacle detection result
```

은 늦게 와도 쓸모가 없을 수 있다.

---

# 12. Hard / Firm / Soft 비교

```text
Hard
→ Deadline miss = unacceptable

Firm
→ 늦은 결과는 쓸모 없음

Soft
→ 늦어도 어느 정도 허용
```

정도로 이해하면 된다.

---

# 13. Robot 내부의 여러 Timing Requirement

예:

```text
Motor Current Control
→ 매우 빠르고 strict

Joint Control
→ strict

State Estimation
→ high frequency / low jitter 중요

SLAM
→ real-time에 가까운 processing 필요

Object Detection
→ low latency 중요

Logging
→ 상대적으로 덜 strict
```

모든 software가 같은 real-time requirement를 가지는 것은 아니다.

---

# 14. Scheduler

CPU에는 실행하고 싶은 process/thread가 여러 개 있다.

```text
ROS 2 Node
FAST-LIO2
Camera
Nav2
Logger
Network
```

하지만 CPU core 수는 제한되어 있다.

어떤 thread를 언제 실행할지 결정하는 역할이:

```text
Scheduler
```

이다.

---

# 15. Linux Scheduler

Linux kernel scheduler가:

```text
Which task?
Which CPU?
How long?
```

을 결정한다.

개념:

```text
Ready Tasks
├── Task A
├── Task B
├── Task C
└── Task D
      │
      ▼
Scheduler
      │
      ▼
CPU
```

---

# 16. Preemption

현재 CPU에서 실행 중인 task를 중간에 멈추고
더 중요한 task를 실행하는 것을:

```text
Preemption
```

이라고 한다.

예:

```text
Low-priority Task
      │
      X interrupted
      │
High-priority Task
```

---

# 17. Preemptive Scheduling

Real-Time system에서는 중요한 task가 준비되면
낮은 priority task를 빠르게 밀어내고 실행할 수 있어야 한다.

```text
Low Priority
───────X

High Priority
       └──────► Run
```

---

# 18. Priority

Task마다 중요도를 나타내는 priority를 둘 수 있다.

예:

```text
Motor Control
Priority High

Logger
Priority Low
```

CPU contention이 발생하면 high-priority task를 먼저 실행하도록 할 수 있다.

---

# 19. Linux의 일반 Process Priority

일반 Linux scheduling에서는:

```text
nice
```

값을 볼 수 있다.

예:

```bash
nice
renice
```

일반적으로 nice 값이 낮을수록 CPU scheduling에서 더 우호적인 priority를 받을 수 있다.

하지만 이것은 hard real-time priority와는 다르다.

---

# 20. Real-Time Scheduling Policy

Linux에는 real-time scheduling policy도 있다.

대표적으로:

```text
SCHED_FIFO
SCHED_RR
```

가 있다.

---

# 21. SCHED_FIFO

FIFO:

```text
First In First Out
```

real-time priority 기반 scheduling 방식이다.

높은 priority task가 실행 가능하면
일반 task보다 우선 실행될 수 있다.

---

# 22. SCHED_RR

Round Robin 방식으로
같은 real-time priority의 task들 사이에서 time slice를 나눌 수 있다.

---

# 23. Real-Time Priority는 위험할 수도 있다

높은 real-time priority의 task가 CPU를 계속 점유하면:

```text
SSH
Shell
System Service
```

같은 중요한 process가 실행되지 못할 수 있다.

따라서 무작정 최고 priority를 주면 안 된다.

---

# 24. CPU Starvation

어떤 task가 CPU를 계속 차지해서 다른 task가 실행되지 못하는 상황:

```text
Starvation
```

이다.

예:

```text
RT Task
100% CPU
     │
     X
Logger / SSH / Other Tasks
```

---

# 25. Context Switch

CPU가:

```text
Task A
```

에서:

```text
Task B
```

로 실행 대상을 바꾸는 것을:

```text
Context Switch
```

라고 한다.

---

# 26. Context에는 무엇이 있을까?

예:

```text
Registers
Program Counter
Stack State
Scheduling State
```

등 현재 task 실행 정보를 저장하고 복원해야 한다.

따라서 context switch도 공짜가 아니다.

---

# 27. 너무 많은 Thread 문제

Thread가 많으면 parallelism이 증가할 수 있지만:

```text
Context Switch
Cache Miss
Lock
Scheduling Overhead
```

도 늘어난다.

즉:

```text
More Threads
≠
Always Better
```

이다.

---

# 28. Scheduling Latency

높은 priority task가 실행 가능해진 순간부터
실제로 CPU를 얻기까지 걸리는 시간이 있다.

이를:

```text
Scheduling Latency
```

라고 볼 수 있다.

Real-Time에서는 이 latency의 worst case가 중요하다.

---

# 29. Interrupt Latency

Chapter 14에서 interrupt를 배웠다.

Hardware가 interrupt를 발생시킨 순간부터
CPU가 해당 interrupt를 처리하기 시작할 때까지 시간이 걸릴 수 있다.

```text
Hardware Interrupt
      │
      │ delay
      ▼
Interrupt Handler
```

이 delay가:

```text
Interrupt Latency
```

다.

---

# 30. Sensor Timing과 연결

예를 들어 IMU가 data ready interrupt를 보냈다.

```text
IMU
 ↓
Interrupt
 ↓
Kernel
 ↓
Driver
 ↓
ROS
 ↓
Estimator
```

각 layer의 latency와 jitter가 합쳐진다.

---

# 31. End-to-End Latency

전체 robot pipeline:

```text
Sensor
  ↓
Driver
  ↓
ROS 2
  ↓
Estimator
  ↓
Controller
  ↓
Motor Command
```

이 전체 delay를 봐야 한다.

한 component만 빠르다고 전체 system이 real-time인 것은 아니다.

---

# 32. Priority Inversion

Real-Time system에서 매우 중요한 문제다.

세 task가 있다고 하자.

```text
High Priority: H
Medium Priority: M
Low Priority: L
```

L이 lock을 가지고 있다.

그 순간 H가 같은 lock을 필요로 한다.

```text
L holds lock
     │
H waits
```

그런데 M이 실행되면 L이 lock을 풀 기회를 못 받을 수 있다.

```text
H waiting
M running
L blocked from CPU
```

결국 high priority H가 low priority L 때문에 늦어진다.

이를:

```text
Priority Inversion
```

이라고 한다.

---

# 33. Priority Inheritance

Priority inversion을 완화하기 위해:

```text
Priority Inheritance
```

를 사용할 수 있다.

L이 H가 필요한 lock을 가지고 있다면
일시적으로 L의 priority를 높여 빨리 lock을 release하게 한다.

```text
H waits for L
      ↓
L temporarily gets high priority
      ↓
L releases lock
      ↓
H runs
```

---

# 34. Mutex

Thread synchronization에서:

```text
Mutex
```

를 자주 사용한다.

공유 resource를 동시에 여러 thread가 수정하지 못하게 보호한다.

```text
Shared Data
     │
     ▼
   Mutex
   /   \
Thread A
Thread B
```

---

# 35. Lock Contention

여러 thread가 같은 lock을 자주 기다리는 상태:

```text
Lock Contention
```

이다.

Parallel program인데 lock contention이 심하면
실제로는 거의 serial하게 동작할 수 있다.

---

# 36. Deadlock

두 task가 서로가 가진 resource를 기다려
영원히 진행하지 못하는 상황:

```text
Deadlock
```

예:

```text
Task A holds Lock 1
Task A waits Lock 2

Task B holds Lock 2
Task B waits Lock 1
```

둘 다 진행하지 못한다.

---

# 37. Real-Time에서는 Lock 설계가 중요하다

긴 critical section이나 blocking lock은
worst-case latency를 증가시킨다.

따라서 real-time code에서는:

```text
Lock duration
Blocking
Priority inversion
```

을 신중하게 본다.

---

# 38. CPU Affinity

Process/thread를 특정 CPU core에 실행하도록 제한할 수 있다.

이를:

```text
CPU Affinity
```

라고 한다.

Linux tool:

```bash
taskset
```

---

# 39. CPU Affinity Example

예:

```text
CPU 0
→ System / Interrupt

CPU 1
→ Control

CPU 2-3
→ SLAM
```

처럼 역할을 나눌 수 있다.

실제 최적 구성은 profiling이 필요하다.

---

# 40. 왜 Affinity를 사용할까?

Scheduler가 thread를 core 사이에서 이동시키면:

```text
Cache locality 감소
Timing variation
```

이 생길 수 있다.

특정 critical thread를 특정 core에 고정하면
timing predictability를 높이는 데 도움이 될 수 있다.

---

# 41. 하지만 Affinity가 항상 좋은 것은 아니다

잘못 고정하면:

```text
Core 1 = overloaded
Core 2 = idle
```

같은 상황이 발생할 수 있다.

따라서:

```text
Measure first
```

가 중요하다.

---

# 42. IRQ Affinity

Hardware interrupt도 특정 CPU에서 처리될 수 있다.

예:

```text
LiDAR NIC IRQ
→ CPU 0

Camera IRQ
→ CPU 1
```

interrupt가 critical application과 같은 core를 과도하게 사용하면
jitter를 증가시킬 수 있다.

---

# 43. `/proc/interrupts`

확인:

```bash
cat /proc/interrupts
```

어떤 interrupt가 어떤 CPU에서 처리되고 있는지 볼 수 있다.

---

# 44. PREEMPT_RT

Standard Linux는 general-purpose OS다.

Hard real-time을 위해 만들어진 것은 아니다.

Linux를 더 preemptible하게 만들어
real-time latency를 개선하는 patch/configuration 계열이:

```text
PREEMPT_RT
```

다.

---

# 45. PREEMPT_RT의 목적

Kernel 내부에서 긴 non-preemptible 구간을 줄이고
높은 priority real-time task가 더 빠르게 실행될 수 있도록 한다.

목표:

```text
Lower worst-case latency
Lower jitter
More predictable scheduling
```

이다.

---

# 46. PREEMPT_RT를 사용하면 Linux가 MCU처럼 되나?

아니다.

PREEMPT_RT는 Linux real-time capability를 크게 개선할 수 있지만:

```text
Complex OS
Memory
Driver
Interrupt
Background Services
```

가 여전히 존재한다.

안전-critical hard real-time control에서는 별도 MCU/RTOS를 사용하는 이유가 있다.

---

# 47. General Linux vs RT Linux

개념적으로:

```text
General Linux

Throughput
Fairness
General workloads
```

에 최적화.

```text
RT Linux

Predictable latency
Priority handling
Deadline-sensitive tasks
```

를 더 중요하게 본다.

---

# 48. RTOS

RTOS:

```text
Real-Time Operating System
```

이다.

예:

```text
FreeRTOS
Zephyr
VxWorks
QNX
```

등이 있다.

---

# 49. RTOS 특징

일반적으로:

```text
Small
Deterministic
Low latency
Priority-based scheduling
```

특성을 강하게 고려한다.

MCU에서 많이 사용된다.

---

# 50. Jetson과 MCU 역할 분리

Vision60 같은 robot에서 conceptually:

```text
Jetson
├── Perception
├── SLAM
├── Navigation
├── AI
└── High-Level Planning

MCU
├── Joint Control
├── Motor Timing
├── Safety
└── Low-Level Control
```

처럼 역할을 나눌 수 있다.

---

# 51. 왜 Motor Control을 Jetson에서만 하지 않을까?

Jetson Linux에서는:

```text
Scheduler jitter
Background process
Interrupt
Thermal state
Memory pressure
```

등이 timing에 영향을 줄 수 있다.

Motor control은 훨씬 deterministic한 timing이 필요할 수 있다.

---

# 52. High-Level vs Low-Level Control

High-Level:

```text
Walk forward
Turn left
Go to waypoint
```

Low-Level:

```text
Joint torque
Joint position
Current control
PWM
```

이다.

---

# 53. Control Frequency

Low-level control은 매우 높은 frequency를 사용할 수 있다.

예:

```text
1 kHz
```

이면:

```text
1 ms period
```

이다.

이 정도 timing에서는 수 ms의 jitter도 큰 문제다.

---

# 54. SLAM Frequency

FAST-LIO2 output이:

```text
100 Hz
```

라고 하더라도 motor current loop의 1 kHz real-time requirement와는 성격이 다르다.

SLAM은 high-frequency estimation이지만
hard real-time actuator control과 같은 요구는 아닐 수 있다.

---

# 55. Sensor Frequency vs Processing Frequency

예:

```text
IMU
200 Hz

LiDAR
10 Hz

Odometry output
100 Hz
```

각 component frequency가 다를 수 있다.

Real-time system에서는 이 data flow의 timing 관계를 이해해야 한다.

---

# 56. Deadline Miss

작업이 정해진 시간보다 늦게 끝나는 것:

```text
Deadline Miss
```

예:

```text
Period = 10 ms

Execution = 15 ms
```

이면 deadline을 놓친다.

---

# 57. Overrun

Periodic task가 다음 period 시작 전에 끝나지 않는 것을:

```text
Overrun
```

이라고 볼 수 있다.

```text
Cycle 1
|---------------|

Cycle 2 시작
          ↑
Cycle 1 아직 실행 중
```

---

# 58. Overrun이 계속되면

```text
Queue grows
Latency grows
Old data processed
```

문제가 발생한다.

Chapter 12에서 배운 producer-consumer 문제와 연결된다.

---

# 59. Drop vs Queue

Real-time sensor pipeline에서는 두 전략이 있을 수 있다.

```text
Queue all
```

장점:

```text
Data loss 적음
```

단점:

```text
Latency 증가
```

또는:

```text
Drop old data
```

장점:

```text
Latest state 유지
```

단점:

```text
Data loss
```

application 요구에 따라 선택한다.

---

# 60. Watchdog

System이나 task가 정상적으로 동작하는지 감시하는 mechanism:

```text
Watchdog
```

이다.

---

# 61. Hardware Watchdog

MCU나 SoC에 hardware watchdog timer가 있을 수 있다.

Application이 주기적으로:

```text
"I'm alive"
```

신호를 보내야 한다.

정해진 시간 동안 신호가 없으면:

```text
Reset
```

할 수 있다.

---

# 62. Software Watchdog

Software process가 다른 process의 heartbeat를 감시할 수도 있다.

예:

```text
FAST-LIO2
   │
Heartbeat
   ▼
Supervisor
```

heartbeat가 끊기면 restart할 수 있다.

---

# 63. Watchdog이 해결책은 아니다

무조건 restart만 하면 원인이 숨겨질 수 있다.

좋은 구조:

```text
Failure detected
      ↓
Log reason
      ↓
Safe state
      ↓
Restart if appropriate
```

이다.

---

# 64. Heartbeat

Component가 살아 있음을 주기적으로 알리는 signal이다.

예:

```text
Node A
 │
 │ heartbeat 10 Hz
 ▼
Monitor
```

heartbeat가 일정 시간 없으면 failure로 판단한다.

---

# 65. ROS 2 Deadline QoS

ROS 2 QoS에는:

```text
Deadline
```

관련 설정도 있다.

예:

> Publisher가 기대되는 주기 안에 데이터를 제공해야 한다.

는 요구를 표현할 수 있다.

Deadline miss event를 감지할 수도 있다.

---

# 66. ROS 2 Liveliness

QoS의:

```text
Liveliness
```

는 publisher가 여전히 살아 있는지 판단하는 데 사용할 수 있다.

Distributed robot system의 health monitoring과 관련된다.

---

# 67. ROS 2와 Real-Time

ROS 2 자체가:

```text
Hard Real-Time Guaranteed
```

인 것은 아니다.

하지만:

```text
DDS QoS
Executors
Allocator
Intra-process
RT scheduling
PREEMPT_RT
```

등을 조합해 real-time 성능을 개선할 수 있다.

---

# 68. Executor와 Timing

Chapter 6에서 executor를 배웠다.

```text
IMU Callback
LiDAR Callback
Timer Callback
```

이 하나의 executor에서 실행될 수 있다.

---

# 69. SingleThreadedExecutor 문제

예:

```text
LiDAR callback = 40 ms

IMU callback period = 5 ms
```

SingleThreadedExecutor라면 LiDAR callback이 길게 실행되는 동안
IMU callback이 지연될 수 있다.

---

# 70. MultiThreadedExecutor

여러 callback을 병렬 처리할 수 있다.

```text
Thread 1
→ LiDAR

Thread 2
→ IMU
```

하지만:

```text
Lock
Race condition
Scheduling jitter
```

을 고려해야 한다.

---

# 71. Callback Group

ROS 2에서는 callback execution 관계를 제어하기 위해
callback group을 사용할 수 있다.

예:

```text
Mutually Exclusive
Reentrant
```

개념을 통해 동시에 실행 가능한 callback을 제어할 수 있다.

---

# 72. Long Callback

Real-time-ish node에서는 callback 안에서 너무 긴 blocking 작업을 하면 좋지 않을 수 있다.

예:

```cpp
void imu_cbk(...)
{
    heavy_file_write();
    network_request();
    sleep(...);
}
```

IMU processing timing을 망칠 수 있다.

---

# 73. Blocking Operation

다음은 thread를 오래 막을 수 있다.

```text
Disk I/O
Network request
Lock wait
Sleep
Large memory allocation
```

critical loop 안에서는 신중해야 한다.

---

# 74. Dynamic Memory Allocation

`malloc`, `new` 등 dynamic allocation은
실행 시간이 항상 일정하다고 보장하기 어렵다.

Hard real-time code에서는 runtime allocation을 줄이고
pre-allocation을 사용하는 경우가 많다.

---

# 75. Page Fault

Program이 접근한 memory page가 즉시 physical memory에 준비되어 있지 않을 때
page fault가 발생할 수 있다.

이는 latency spike를 만들 수 있다.

---

# 76. Swap

RAM이 부족하면 memory page가 storage로 swap될 수 있다.

```text
RAM
 ↓
SSD
```

Storage는 RAM보다 훨씬 느리다.

Real-time workload에서 swap은 큰 jitter 원인이 될 수 있다.

---

# 77. Memory Locking

Real-time application에서는 memory가 swap되지 않도록:

```text
mlockall()
```

같은 mechanism을 사용하는 경우가 있다.

이 역시 충분히 이해하고 사용해야 한다.

---

# 78. Garbage Collection

Python, Java 등의 runtime에서는 garbage collection이 timing에 영향을 줄 수 있다.

예:

```text
Normal execution
1 ms
1 ms
1 ms
GC pause
30 ms
```

Hard real-time에서는 이런 unpredictable pause가 문제다.

---

# 79. C++가 Real-Time에서 많이 쓰이는 이유

C++는:

```text
Memory control
No mandatory GC
Low-level access
Predictable execution 설계 가능
```

이라는 장점이 있다.

물론 C++라고 자동으로 real-time이 되는 것은 아니다.

---

# 80. Rust와 Real-Time

Rust도:

```text
No GC
Memory safety
Low-level control
```

특성 때문에 embedded/real-time 영역에서 사용할 수 있다.

하지만 library/runtime/OS 구조까지 함께 봐야 한다.

---

# 81. Python은 Real-Time이 불가능한가?

그렇게 단정할 수는 없다.

하지만 CPython은:

```text
Interpreter overhead
GC
GIL
OS scheduling
```

등 때문에 strict hard real-time control에 일반적으로 적합하지 않다.

High-level control, orchestration, tooling에는 매우 유용하다.

---

# 82. GIL

CPython에는:

```text
Global Interpreter Lock
```

이 있다.

하나의 process에서 Python bytecode 실행이 여러 thread에서 완전히 병렬로 실행되지 않는 제약과 관련된다.

I/O나 native extension에서는 상황이 다를 수 있다.

---

# 83. Real-Time System은 전체 Chain 문제다

C++로 작성했다고 real-time이 아니다.

예:

```text
C++ Control Loop
   ↓
General Linux
   ↓
Non-RT Driver
   ↓
Wi-Fi
```

라면 timing jitter가 생길 수 있다.

전체 stack을 봐야 한다.

---

# 84. Ethernet과 Real-Time

일반 Ethernet은 high bandwidth지만
strict deterministic timing을 자동으로 보장하지는 않는다.

Industrial real-time networking에서는:

```text
EtherCAT
TSN
PROFINET IRT
```

같은 기술도 존재한다.

---

# 85. CAN과 Timing

CAN은 priority-based arbitration을 사용한다.

높은 priority message가 먼저 bus를 사용할 수 있다.

그래서 embedded control system에서 predictable communication을 설계하기 좋다.

하지만 bus load가 너무 높으면 latency가 증가할 수 있다.

---

# 86. Bus Utilization

CAN bus가 거의 100% 사용 중이라면
새 message가 기다려야 한다.

따라서:

```text
Bus Load
Message Rate
Priority
```

를 설계해야 한다.

---

# 87. Time Synchronization과 Real-Time은 다르다

Chapter 11:

```text
Time Synchronization
→ 여러 장치의 clock을 맞추기
```

Chapter 15:

```text
Real-Time
→ 작업이 deadline 안에 실행되도록 하기
```

이다.

Clock이 정확히 맞아 있어도 task가 늦게 실행될 수 있다.

---

# 88. PTP가 있다고 Real-Time은 아니다

PTP:

```text
Clock sync
```

PREEMPT_RT:

```text
Scheduling latency 개선
```

이 둘은 다른 문제를 해결한다.

---

# 89. Real-Time Measurement

Timing은 실제로 측정해야 한다.

예:

```text
Expected Period = 10 ms
```

실제:

```text
9.9
10.1
10.0
12.3
9.8
```

이 distribution을 측정한다.

---

# 90. Average만 보면 안 된다

Real-time에서는:

```text
Average
Median
P95
P99
Maximum
```

을 함께 본다.

특히:

```text
Worst Case
```

가 중요하다.

---

# 91. Worst-Case Execution Time

WCET:

```text
Worst-Case Execution Time
```

이다.

특정 task가 가장 오래 걸릴 때의 실행 시간을 분석하는 개념이다.

Hard real-time에서는 중요하다.

---

# 92. Worst-Case Response Time

Task가 ready된 후 실제로 완료될 때까지의 최악의 시간을 볼 수도 있다.

```text
Scheduling wait
+
Execution
+
Blocking
=
Response Time
```

---

# 93. cyclictest

Linux real-time latency를 평가할 때 많이 사용하는 tool:

```text
cyclictest
```

가 있다.

PREEMPT_RT system benchmark에서 자주 사용된다.

---

# 94. cyclictest 개념

Periodic thread를 깨워서:

```text
Expected Wakeup Time
vs
Actual Wakeup Time
```

차이를 측정한다.

즉 scheduler latency/jitter를 관찰할 수 있다.

---

# 95. `chrt`

Linux에서 scheduling policy/real-time priority를 확인하거나 설정할 때:

```bash
chrt
```

를 사용할 수 있다.

무작정 높은 priority를 설정하지 않는다.

---

# 96. `taskset`

CPU affinity:

```bash
taskset
```

예:

```bash
taskset -c 2 ./my_program
```

은 program을 특정 CPU에 묶는 용도로 사용할 수 있다.

---

# 97. `ps`로 Scheduler 정보 보기

환경에 따라:

```bash
ps -eLo pid,tid,cls,rtprio,pri,psr,comm
```

같은 방식으로 thread scheduling class와 CPU를 확인할 수 있다.

---

# 98. `/proc/<pid>/status`

Process의 여러 상태 정보:

```bash
cat /proc/<pid>/status
```

에서 CPU/memory 관련 정보를 볼 수 있다.

---

# 99. Real-Time Debugging 순서

Timing 문제가 있다면:

```text
1. Required period/deadline 정의
       ↓
2. Actual latency 측정
       ↓
3. Jitter 분포 확인
       ↓
4. CPU utilization 확인
       ↓
5. Thread/blocking 확인
       ↓
6. Interrupt load 확인
       ↓
7. Scheduling policy 확인
       ↓
8. Affinity/priority 검토
       ↓
9. PREEMPT_RT 필요성 검토
```

---

# 100. Robot Example

Vision60에서:

```text
IMU
200 Hz
→ 5 ms period

FAST-LIO2
100 Hz
→ 10 ms output

Joint Controller
1 kHz
→ 1 ms period
```

라고 가정해보자.

각 loop는 요구 timing이 다르다.

---

# 101. Jetson에 모든 것을 넣으면?

```text
SLAM
Camera AI
Navigation
Joint Control
Logging
```

모두 general Linux 위에서 경쟁한다.

갑자기 camera AI가 큰 GPU/CPU workload를 발생시키면
critical control timing에 영향을 줄 수 있다.

---

# 102. 역할 분리의 이유

그래서:

```text
Jetson
→ Heavy high-level compute

MCU / RT Controller
→ Deterministic low-level control
```

구조가 실용적이다.

---

# 103. Safety Layer

High-level computer가 멈춰도
low-level controller가 안전 동작을 수행할 수 있어야 한다.

예:

```text
Jetson heartbeat lost
       ↓
MCU detects timeout
       ↓
Stop / Safe state
```

---

# 104. Command Timeout

Robot control에서는:

```text
Last command received > 100 ms ago
```

같은 조건이면 command를 더 이상 신뢰하지 않을 수 있다.

이를 timeout 기반 fail-safe로 설계할 수 있다.

---

# 105. Stale Command

Network나 Jetson이 멈췄는데
마지막 velocity command를 계속 적용하면 위험하다.

그래서:

```text
Command timestamp
+
Timeout
```

가 중요하다.

---

# 106. E-stop과 Real-Time

Emergency Stop은 software high-level stack보다
더 직접적이고 신뢰도 높은 safety path로 설계되는 경우가 많다.

```text
E-stop
  ↓
Safety Controller
  ↓
Actuator Disable
```

정확한 Vision60 구조는 제조사 hardware 설계를 확인해야 한다.

---

# 107. Performance vs Determinism

GPU를 사용하면 throughput은 크게 높아질 수 있다.

하지만:

```text
Maximum Throughput
```

과:

```text
Predictable Worst-Case Latency
```

는 다른 목표다.

---

# 108. Real-Time GPU?

GPU도 real-time workload에 사용할 수 있지만
resource scheduling과 kernel execution time을 잘 관리해야 한다.

일반적인 deep learning workload는 timing variation이 있을 수 있다.

---

# 109. Resource Isolation

Critical workload가 다른 task의 영향을 덜 받게 하기 위해:

```text
CPU Affinity
Priority
Memory Isolation
Container Resource Limits
Separate Computer
```

등을 사용할 수 있다.

---

# 110. Docker와 Real-Time

Container 자체는 VM처럼 별도 kernel을 가지지 않는다.

따라서 host Linux scheduler의 영향을 받는다.

```text
Container RT App
       │
       ▼
Host Kernel Scheduler
```

이다.

---

# 111. Container가 Timing을 자동 보장하지 않는다

Docker를 사용한다고 real-time이 되는 것이 아니다.

필요하면:

```text
Scheduling capability
CPU affinity
Resource limit
Host RT kernel
```

등을 고려해야 한다.

---

# 112. CPU Resource Limit

Container에 CPU limit을 걸면 resource isolation에 도움이 될 수 있지만
critical task가 필요한 CPU를 못 받게 만들 수도 있다.

실제 요구사항을 기준으로 설정해야 한다.

---

# 113. Real-Time Logging

Critical control thread가 직접 큰 log를 disk에 쓰면
I/O latency 때문에 timing에 영향을 줄 수 있다.

더 좋은 구조:

```text
Critical Thread
      │
      ▼
Lock-free / bounded queue
      │
      ▼
Low-priority Logger Thread
      │
      ▼
Disk
```

같은 방식이 가능하다.

---

# 114. Bounded Queue

Queue 크기를 제한하면:

```text
Memory 무한 증가
```

를 방지할 수 있다.

Queue가 가득 찼을 때:

```text
Drop?
Block?
Overwrite?
```

정책을 미리 정해야 한다.

---

# 115. Lock-Free

Lock을 최소화하는 data structure를:

```text
Lock-Free
```

라고 부를 수 있다.

Real-time system에서 lock blocking을 줄이기 위해 고려할 수 있다.

하지만 구현이 복잡하므로 무조건 사용할 필요는 없다.

---

# 116. Real-Time Design 원칙

Critical loop에서는 가능한 한:

```text
No unpredictable blocking
No long file I/O
No network waits
No unbounded queue
No unnecessary allocation
Short lock duration
```

를 목표로 한다.

---

# 117. Vision60 Mental Model

```text
                    Vision60

         ┌────────────────────────┐
         │ Jetson / Xavier        │
         │                        │
         │ ROS 2                  │
         │ FAST-LIO2              │
         │ Navigation             │
         │ Vision AI              │
         │                        │
         │ Soft / Firm Real-Time  │
         └───────────┬────────────┘
                     │
                     │ Command / State
                     ▼
         ┌────────────────────────┐
         │ MCU / Controller       │
         │                        │
         │ Joint Control          │
         │ Motor Timing           │
         │ Safety                 │
         │                        │
         │ Harder Real-Time       │
         └───────────┬────────────┘
                     │
                     ▼
                   Motors
```

---

# 118. Timing Mental Model

전체 robot timing:

```text
Sensor
   │
   │ Measurement Time
   ▼
Driver
   │
   │ Interrupt / I/O latency
   ▼
ROS 2
   │
   │ Queue / Executor latency
   ▼
Algorithm
   │
   │ Compute latency
   ▼
Controller
   │
   │ Scheduling latency
   ▼
Actuator
```

전체 합이 deadline 안에 들어와야 한다.

---

# 119. Mini Practice 1

Frequency를 period로 바꿔본다.

```text
1000 Hz = ?
200 Hz  = ?
100 Hz  = ?
50 Hz   = ?
10 Hz   = ?
```

답:

```text
1000 Hz = 1 ms
200 Hz  = 5 ms
100 Hz  = 10 ms
50 Hz   = 20 ms
10 Hz   = 100 ms
```

---

# 120. Mini Practice 2

다음 latency:

```text
9.8
10.0
9.9
10.1
45.0 ms
```

평균만 보고:

```text
"약 17 ms니까 괜찮다"
```

라고 판단하면 안 된다.

질문:

```text
왜 45 ms가 발생했는가?
Deadline은 얼마인가?
```

를 봐야 한다.

---

# 121. Mini Practice 3

Jetson에서:

```bash
cat /proc/interrupts
```

를 실행한다.

FAST-LIO2/Camera workload 전후로
어떤 interrupt count가 빠르게 증가하는지 관찰한다.

---

# 122. Mini Practice 4

Process 확인:

```bash
ps -eLo pid,tid,psr,pri,rtprio,comm | head -n 30
```

각 thread가 어느 CPU에서 실행되는지 살펴본다.

---

# 123. Mini Practice 5

환경이 허용한다면 test program을:

```bash
taskset -c 2 <command>
```

로 실행해보고 CPU affinity 개념을 확인한다.

Production robot에서는 이유 없이 affinity를 변경하지 않는다.

---

# 124. Mini Practice 6

Vision60 stack의 component를 직접 분류한다.

예:

| Component | Timing Type |
|---|---|
| Motor current control | Hard/Strict RT |
| Joint control | RT |
| IMU processing | Low-latency |
| FAST-LIO2 | Soft/Firm RT |
| Object detection | Soft/Firm RT |
| Logging | Non-critical |

실제 요구사항은 system specification을 기준으로 정의한다.

---

# 125. 반드시 구분할 것

```text
Fast
≠
Real-Time

Frequency
≠
Deterministic Period

Latency
≠
Jitter

Average Latency
≠
Worst-Case Latency

Priority
≠
CPU Affinity

Interrupt
≠
Thread

Time Synchronization
≠
Real-Time Scheduling

PREEMPT_RT
≠
RTOS

Container
≠
Real-Time Isolation

High Priority
≠
Always Better
```

---

# 126. Chapter 15 핵심

Real-Time system의 핵심 질문은:

```text
"얼마나 빨리?"
```

뿐만 아니라:

```text
"최악의 경우에도 언제까지?"
```

이다.

즉:

```text
Performance
+
Predictability
+
Deadline
```

를 함께 본다.

---

# 127. Robot Real-Time Mental Model

```text
Task
  │
  ▼
Required Frequency
  │
  ▼
Period
  │
  ▼
Deadline
  │
  ▼
Scheduler / Priority
  │
  ▼
Actual Execution
  │
  ├── Latency
  ├── Jitter
  └── Worst Case
  │
  ▼
Deadline Met?
```

---

# 128. 왜 이 Chapter가 Edge Computing에서 중요한가?

Edge computer는 sensor와 actuator 가까이에서 동작한다.

따라서 cloud server처럼 단순히:

```text
"언젠가 계산이 끝나면 된다."
```

가 아니다.

Robot은 계속 움직이므로:

```text
Correct Result
+
Correct Time
```

이 둘이 모두 중요하다.

---

# Next Chapter

## Chapter 16. Embedded Security

다음 Chapter에서는 robot edge computer를 network에 연결할 때 필요한 security를 다룬다.

```text
SSH Key
Password
User / Root
Firewall
Ports
Secrets
Certificates
TLS
Secure Boot
Disk Encryption
Least Privilege
Container Security
Software Updates
```

특히:

```text
Company Wi-Fi
      │
      ▼
Jetson
      │
      ▼
Robot Internal Network
```

구조에서 왜 security가 중요해지는지 살펴본다.

그리고:

```text
"SSH가 된다는 것은 누가 어디까지 들어올 수 있다는 뜻인가?"

"GitHub token이나 AWS key를 robot 안에 그냥 저장해도 되는가?"

"ROS 2 network를 회사 network에 그대로 노출해도 되는가?"
```

같은 실제 edge deployment 문제를 다룬다.