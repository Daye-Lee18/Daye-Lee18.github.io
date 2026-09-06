---
title: "Chapter 1. Computer Hardware Basics"
importance: 2
---
> **Goal:** Jetson을 배우기 전에 CPU, GPU, RAM, Storage가 각각 무슨 일을 하는지 이해하고,
> 이것들이 실제 로봇에서 어떻게 사용되는지 연결한다.

---

## 1. 로봇도 결국 하나의 컴퓨터다

Vision60 같은 로봇 안에는 여러 컴퓨터와 MCU가 들어간다.

예를 들어 전체 구조를 아주 단순화하면 다음과 같다.

```text
LiDAR ─────┐
Camera ────┤
IMU ───────┤
           ▼
     ┌─────────────┐
     │ Edge Computer│
     │ Xavier / Orin│
     └──────┬──────┘
            │
     Perception / SLAM
     Navigation / Control
            │
            ▼
           MCU
            │
            ▼
         Motors
```

센서에서 들어온 데이터를 Edge Computer가 처리하고,
그 결과를 이용하여 로봇의 위치를 추정하거나 움직임을 결정한다.

Vision60의 NVIDIA Xavier나 Jetson Orin 역시 본질적으로는 **Linux가 돌아가는 컴퓨터**다.

---

# 2. CPU

CPU는 **Central Processing Unit**의 약자다.

컴퓨터에서 일반적인 프로그램의 명령을 실행하는 핵심 프로세서다.

예를 들어 다음과 같은 작업을 한다.

- Linux 운영체제 실행
- ROS 2 node 실행
- C++ / Rust / Python 코드 실행
- 네트워크 통신
- 파일 읽기/쓰기
- 센서 driver 실행
- 프로그램의 전체적인 흐름 제어

예를 들어 ROS 2에서 다음과 같은 코드가 있다고 하자.

```cpp
while (rclcpp::ok()) {
    read_sensor();
    process_data();
    publish_data();
}
```

이 프로그램의 명령을 실제로 하나씩 수행하는 주체가 CPU다.

---

## 2.1 CPU Core

CPU 내부에는 여러 개의 **Core**가 존재할 수 있다.

```text
CPU
├── Core 0
├── Core 1
├── Core 2
├── Core 3
├── Core 4
├── Core 5
├── Core 6
└── Core 7
```

각 core는 독립적으로 명령을 실행할 수 있기 때문에 여러 작업을 동시에 처리할 수 있다.

로봇에서는 예를 들어:

```text
Core 0 → ROS 2 communication
Core 1 → LiDAR driver
Core 2 → IMU processing
Core 3 → SLAM
Core 4 → Navigation
...
```

처럼 여러 프로세스/thread가 CPU 자원을 나눠 사용할 수 있다.

실제로 어떤 core가 어떤 프로그램만 담당하도록 고정되는 것은 아니며,
Linux scheduler가 실행할 작업을 여러 CPU core에 배분한다.

---

# 3. GPU

GPU는 **Graphics Processing Unit**의 약자다.

원래 그래픽 계산을 위해 만들어졌지만,
현재는 **같은 종류의 계산을 매우 많이 병렬로 처리하는 작업**에도 사용된다.

CPU와 GPU의 핵심적인 차이는 다음과 같이 생각할 수 있다.

```text
CPU

강력한 작업자 몇 명
↓
복잡하고 서로 다른 작업을 처리하는 데 강함


GPU

비교적 단순한 작업자 수백~수천 명
↓
비슷한 계산을 대량으로 동시에 처리하는 데 강함
```

예를 들어 이미지에는 수백만 개의 pixel이 존재한다.

각 pixel 또는 tensor element에 비슷한 계산을 수행해야 한다면 GPU의 병렬 처리 능력을 활용할 수 있다.

---

## 3.1 로봇에서 GPU를 사용하는 곳

대표적으로:

- Object Detection
- Semantic Segmentation
- Depth Estimation
- Neural Network inference
- Reinforcement Learning
- Computer Vision
- CUDA 기반 연산

등이 있다.

예를 들어:

```text
Camera
   ↓
Image
   ↓
GPU
   ↓
Neural Network
   ↓
Person / Vehicle / Obstacle Detection
```

Jetson이 로봇 분야에서 많이 사용되는 중요한 이유 중 하나가
**CPU와 NVIDIA GPU를 하나의 Edge Computer에서 사용할 수 있기 때문**이다.

---

# 4. CPU vs GPU

| CPU | GPU |
|---|---|
| 복잡한 제어 흐름에 강함 | 대규모 병렬 계산에 강함 |
| 비교적 적은 수의 강력한 core | 매우 많은 병렬 연산 unit |
| OS, ROS 2, driver 등 | AI, Vision, CUDA 등 |
| general-purpose processing | parallel processing |

중요한 것은:

> GPU가 CPU보다 무조건 빠른 것이 아니다.

작업 특성에 따라 다르다.

예를 들어 ROS 2 node의 일반적인 제어 흐름이나 파일 처리 등을
GPU로 옮긴다고 해서 빨라지는 것은 아니다.

반대로 거대한 matrix multiplication이나 neural network inference처럼
병렬화하기 좋은 계산은 GPU가 매우 강하다.

---

# 5. RAM

RAM은 **Random Access Memory**다.

현재 실행 중인 프로그램과 데이터를 임시로 저장하는 공간이다.

예를 들어 FAST-LIO2를 실행한다면:

```text
SSD
 │
 │ program load
 ▼
RAM
 ├── FAST-LIO2 program
 ├── incoming LiDAR points
 ├── IMU data
 ├── current map
 └── state estimation data
       │
       ▼
      CPU
```

프로그램을 실행하면 필요한 코드와 데이터가 Storage에서 RAM으로 올라오고,
CPU가 RAM에 있는 데이터를 사용하여 계산한다.

---

## 5.1 RAM은 왜 필요한가?

Storage보다 RAM이 훨씬 빠르기 때문이다.

CPU가 계산할 때마다 SSD에서 데이터를 직접 가져온다면 너무 느리다.

따라서:

```text
Storage
   ↓
  RAM
   ↓
  CPU
```

구조로 사용한다.

---

# 6. Storage

Storage는 데이터를 **장기간 보관하는 공간**이다.

대표적으로:

- SSD
- NVMe SSD
- eMMC
- SD Card

등이 있다.

전원이 꺼져도 데이터가 유지된다.

로봇에서는 다음과 같은 것들이 저장될 수 있다.

```text
Storage
├── Ubuntu
├── ROS 2
├── Robot Software
├── FAST-LIO2
├── Maps
├── rosbag
├── Logs
└── AI Models
```

---

# 7. RAM vs Storage

RAM과 Storage는 자주 혼동되지만 완전히 다른 역할을 한다.

| RAM | Storage |
|---|---|
| 작업 공간 | 장기 저장 공간 |
| 매우 빠름 | RAM보다 느림 |
| 실행 중인 데이터 | 파일/프로그램 저장 |
| 전원을 끄면 데이터 사라짐 | 전원을 꺼도 유지 |
| 예: 32 GB RAM | 예: 1 TB NVMe SSD |

비유하면:

```text
Storage = 책장
RAM     = 책상
CPU     = 책상에서 일하는 사람
```

책장에서 필요한 책을 꺼내 책상 위에 놓고 작업하는 것과 비슷하다.

---

# 8. Jetson의 특징

일반적인 Desktop PC에서는 CPU와 GPU가 별도의 장치인 경우가 많다.

```text
Desktop

CPU ── RAM
 │
PCIe
 │
GPU ── VRAM
```

Jetson은 조금 다르다.

Jetson은 **SoC(System on Chip)** 기반으로 설계되어 있다.

하나의 시스템에 CPU, GPU, memory controller 등 여러 기능이 통합되어 있다.

```text
Jetson SoC
┌──────────────────────────┐
│                          │
│   CPU         GPU        │
│                          │
│      Memory Controller   │
│                          │
│   Video / AI Engines     │
│                          │
└────────────┬─────────────┘
             │
         System RAM
```

특히 Jetson에서는 CPU와 GPU가 **Unified Memory Architecture**를 사용하여
같은 system memory를 공유할 수 있다.

이것은 센서 데이터를 CPU와 GPU 사이에서 처리해야 하는
로봇/AI workload에서 중요한 특징이다.

---

# 9. SoC란?

SoC는 **System on Chip**이다.

기존 컴퓨터에서 여러 chip으로 존재하던 기능을 하나의 chip에 통합한 구조다.

예를 들어:

```text
Traditional PC

CPU
GPU
Memory Controller
Video Processor
I/O Controller
...

       ↓ integration

Jetson SoC

┌────────────────────┐
│ CPU                │
│ GPU                │
│ Memory Controller  │
│ Video Engine       │
│ AI Accelerator     │
│ I/O                │
└────────────────────┘
```

이러한 구조는 크기와 전력 소비가 중요한 로봇, 드론, 자율주행 시스템에 유리하다.

---

# 10. Edge Computing

그렇다면 **Edge Computer**에서 Edge는 무슨 뜻일까?

데이터를 멀리 떨어진 Cloud Server까지 보내서 처리하지 않고,
**데이터가 발생하는 현장 가까이에서 직접 처리하는 것**을 Edge Computing이라고 한다.

예를 들어 로봇이 LiDAR 데이터를 받았다고 하자.

Cloud 방식:

```text
LiDAR
  ↓
Robot
  ↓
Internet
  ↓
Cloud Server
  ↓
SLAM
  ↓
Internet
  ↓
Robot
```

Edge 방식:

```text
LiDAR
  ↓
Jetson
  ↓
SLAM
  ↓
Robot Control
```

로봇에서는 실시간성이 매우 중요하기 때문에 Edge Computing이 특히 중요하다.

네트워크가 끊겨도 로봇이 계속 동작해야 하는 경우가 많기 때문이다.

---

# 11. Vision60에 연결해서 생각하기

Vision60을 예로 들면 전체 구조를 다음처럼 생각할 수 있다.

```text
                Vision60

 LiDAR ──────────────┐
 IMU ────────────────┤
 Camera ─────────────┤
                     ▼
              ┌─────────────┐
              │ Xavier /    │
              │ Jetson Orin │
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
      ROS 2       FAST-LIO2    Perception
        │            │            │
        └────────────┼────────────┘
                     │
              Robot State /
              Motion Command
                     │
                     ▼
                    MCU
                     │
                     ▼
                  Motors
```

여기에서:

- **CPU**: Linux, ROS 2, driver, SLAM의 일반 연산 등
- **GPU**: AI / Vision / CUDA 병렬 연산
- **RAM**: sensor data, map, 실행 중인 프로그램
- **Storage**: Ubuntu, ROS 2, source code, map, rosbag, logs
- **Jetson/Xavier**: 위 요소들을 포함하는 Edge Computer
- **MCU**: 모터와 같은 하드웨어에 가까운 실시간 제어

라고 볼 수 있다.

---

# 12. 오늘의 핵심

이 Chapter에서 가장 중요한 그림은 이것이다.

```text
                ROBOT COMPUTER

Sensors
   │
   ▼
┌────────────────────────────┐
│       Edge Computer        │
│                            │
│  CPU ─── General Programs  │
│  GPU ─── Parallel Compute  │
│  RAM ─── Working Memory    │
│  SSD ─── Persistent Data   │
│                            │
└─────────────┬──────────────┘
              │
              ▼
          Robot Control
```

그리고 Jetson을 공부할 때 계속 다음 질문을 던져야 한다.

> **이 계산은 CPU에서 돌아가는가, GPU에서 돌아가는가?  
> 데이터는 RAM 어디에 존재하는가?  
> 센서 데이터는 어떤 경로로 Jetson까지 들어오는가?**

이 세 가지를 이해하기 시작하면 이후의 CUDA, TensorRT, ROS 2, sensor interface가 서로 연결되기 시작한다.

---

# Next Chapter

## Chapter 2. ARM vs x86

다음 Chapter에서는 다음 질문을 다룬다.

- Jetson은 왜 ARM CPU를 사용하는가?
- 내 Mac/PC의 CPU와 Jetson CPU는 무엇이 다른가?
- x86_64와 aarch64는 무엇인가?
- 왜 어떤 Docker image는 Jetson에서 실행되지 않는가?
- `uname -m`은 무엇을 보여주는가?
- Cross Compilation은 왜 필요한가?

```bash
uname -m
```

Jetson에서 위 명령을 실행했을 때 나오는 `aarch64`가 Chapter 2의 시작점이다.