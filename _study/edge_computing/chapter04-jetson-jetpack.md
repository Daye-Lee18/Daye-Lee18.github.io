---
title: "Chapter 4. NVIDIA Jetson & JetPack"
importance: 5
---

> **Goal:** NVIDIA Jetson이 어떤 하드웨어인지 이해하고,
> JetPack, Ubuntu, L4T, CUDA, cuDNN, TensorRT가 서로 어떤 관계인지 이해한다.
> 또한 Jetson이 왜 로봇용 Edge Computer로 많이 사용되는지 실제 Vision60 관점에서 연결한다.

---

# 1. Jetson은 정확히 무엇인가?

NVIDIA Jetson은 로봇, 드론, 자율주행, AI Edge Computing 등을 위해 만들어진
**ARM 기반의 Embedded AI Computer Platform**이다.

Jetson에는 보통 다음이 통합되어 있다.

```text
Jetson

├── ARM CPU
├── NVIDIA GPU
├── RAM
├── Memory Controller
├── Video Encoder / Decoder
├── Camera Interface
├── PCIe
├── Ethernet
├── USB
└── 기타 I/O
```

즉 Jetson은 단순히 GPU 하나가 아니라:

> CPU + GPU + Memory + I/O가 통합된 하나의 컴퓨터 시스템

이라고 보는 것이 맞다.

---

# 2. 왜 로봇에 Jetson을 많이 사용할까?

로봇은 보통 동시에 여러 종류의 계산이 필요하다.

예를 들어:

```text
LiDAR
Camera
IMU
Joint Encoder
        │
        ▼
     Jetson
        │
        ├── ROS 2
        ├── SLAM
        ├── Object Detection
        ├── Navigation
        └── Robot Control
```

이때 필요한 것은:

```text
General CPU Processing
+
GPU Parallel Processing
+
Low Power
+
Small Size
+
Sensor I/O
```

이다.

Jetson은 이 조건을 한 장치에 통합한 플랫폼이다.

---

# 3. Jetson은 PC와 무엇이 다를까?

일반적인 Desktop PC는:

```text
CPU
 │
 ├── System RAM
 │
 └── PCIe
      │
      ▼
     GPU
      │
      └── VRAM
```

구조인 경우가 많다.

Jetson은 SoC 기반으로:

```text
Jetson SoC

┌──────────────────────────┐
│                          │
│ ARM CPU      NVIDIA GPU  │
│                          │
│ Memory Controller        │
│                          │
│ Video Engines            │
│ AI Accelerators          │
│ I/O Controllers          │
│                          │
└─────────────┬────────────┘
              │
              ▼
         System Memory
```

처럼 여러 기능이 훨씬 긴밀하게 통합되어 있다.

---

# 4. Jetson Module과 Developer Kit

Jetson을 구매할 때 헷갈리는 개념이 있다.

```text
Jetson Module
Jetson Developer Kit
```

둘은 다르다.

---

## 4.1 Jetson Module

실제 computing module이다.

예:

```text
Jetson AGX Orin Module
Jetson Orin NX Module
Jetson Orin Nano Module
```

Module 내부에는:

```text
CPU
GPU
RAM
Storage 일부
```

등이 포함된다.

하지만 module 자체만으로는 일반적으로:

```text
USB port
Ethernet port
HDMI
Power connector
```

등을 직접 사용하기 어렵다.

그래서 carrier board가 필요하다.

---

## 4.2 Developer Kit

Developer Kit은:

```text
Jetson Module
+
Carrier Board
+
Cooling
+
Power Supply
```

등을 묶어 개발자가 바로 사용할 수 있게 만든 제품이다.

구조:

```text
Developer Kit

┌─────────────────────────┐
│ Carrier Board           │
│                         │
│   ┌─────────────────┐   │
│   │ Jetson Module   │   │
│   └─────────────────┘   │
│                         │
│ USB / Ethernet / HDMI   │
│ PCIe / Camera / GPIO    │
└─────────────────────────┘
```

즉:

> Module = 실제 컴퓨팅 핵심
>
> Developer Kit = 개발할 수 있도록 주변 hardware까지 포함한 전체 보드

라고 생각하면 된다.

---

# 5. Carrier Board란?

Carrier Board는 Jetson Module을 실제 장치와 연결해주는 board다.

예:

```text
Jetson Module
      │
      ▼
Carrier Board
      │
      ├── Ethernet
      ├── USB
      ├── PCIe
      ├── M.2
      ├── Camera
      ├── GPIO
      └── Power
```

제품화된 로봇에서는 NVIDIA Developer Kit 그대로 사용하는 대신
custom carrier board를 사용하는 경우도 많다.

이유는:

```text
Size
Power
Connector
Environment
Mechanical Design
```

등을 제품에 맞게 최적화하기 위해서다.

---

# 6. Jetson 제품군

Jetson에는 여러 세대와 등급이 있다.

예:

```text
Jetson Nano
Jetson Xavier NX
Jetson AGX Xavier
Jetson Orin Nano
Jetson Orin NX
Jetson AGX Orin
Jetson AGX Thor
```

대체로:

```text
Nano
  ↓
NX
  ↓
AGX
```

순으로 computing capability와 power budget이 커지는 식으로 볼 수 있다.

하지만 세대마다 구조가 다르기 때문에 단순히 이름만 보고 성능을 판단하면 안 된다.

---

# 7. Xavier, Orin, Thor

Vision60과 연결해서 보면 이 이름들을 자주 만나게 된다.

## Xavier

이전 세대 Jetson platform.

예:

```text
Jetson AGX Xavier
Jetson Xavier NX
```

Vision60 내부 onboard computer로 NVIDIA Xavier가 사용되는 경우가 있다.

---

## Orin

Xavier 다음 세대.

예:

```text
Jetson AGX Orin
Jetson Orin NX
Jetson Orin Nano
```

더 높은 AI compute capability와 향상된 GPU architecture를 제공한다.

로봇에서:

```text
SLAM
Vision
Deep Learning
Planning
```

을 함께 수행하기에 강력하다.

---

## Thor

더 새로운 세대의 고성능 Jetson platform이다.

로봇과 Physical AI workload를 강하게 겨냥한 platform으로 볼 수 있다.

구체적인 성능과 software support는 JetPack 버전과 제품 출시 시점에 따라 확인해야 한다.

실무에서는:

> Jetson 이름만 볼 것이 아니라
> 정확한 Module + JetPack + CUDA 조합을 확인해야 한다.

---

# 8. JetPack이란?

JetPack은 Jetson에서 매우 중요한 software stack이다.

많이 하는 오해:

```text
JetPack = Ubuntu
```

가 아니다.

JetPack은 Jetson을 위한 NVIDIA software stack이다.

대략:

```text
JetPack

├── Jetson Linux
├── Ubuntu Root Filesystem
├── NVIDIA Driver
├── CUDA
├── cuDNN
├── TensorRT
├── Multimedia API
└── Development Tools
```

등을 포함한다.

즉:

> JetPack은 Jetson Hardware를 사용하기 위한 NVIDIA의 전체 software package 묶음

이라고 이해하면 된다.

---

# 9. Jetson Linux와 Ubuntu

Jetson에서는 Ubuntu를 사용하지만,
일반 PC Ubuntu와 완전히 동일한 boot/kernel/driver 환경은 아니다.

NVIDIA는 Jetson용:

```text
Jetson Linux
```

를 제공한다.

과거에는 흔히:

```text
L4T
```

라는 이름을 많이 봤다.

L4T는:

**Linux for Tegra**

의 약자다.

Tegra는 NVIDIA의 embedded SoC 계열 이름이다.

---

# 10. 전체 Software Stack

Jetson의 software stack을 위에서 아래로 보면:

```text
Robot Application

ROS 2
FAST-LIO2
PyTorch
TensorRT

        │
        ▼

CUDA / cuDNN
NVIDIA Libraries

        │
        ▼

Ubuntu User Space

        │
        ▼

Jetson Linux
Linux Kernel
NVIDIA Drivers

        │
        ▼

Jetson Hardware
CPU / GPU / I/O
```

이 구조를 이해하는 것이 중요하다.

---

# 11. JetPack과 Jetson Linux의 관계

대략적으로:

```text
JetPack
   │
   ├── Jetson Linux
   ├── CUDA
   ├── cuDNN
   ├── TensorRT
   └── 기타 SDK
```

라고 보면 된다.

즉 Jetson Linux는 JetPack 구성 요소 중 하나다.

---

# 12. L4T Version

Jetson에서 system 정보를 확인하다 보면:

```text
R36.x
R35.x
```

같은 version을 볼 수 있다.

이것은 Jetson Linux / L4T release 계열을 의미한다.

JetPack version과 L4T version은 서로 연결되어 있다.

예를 들어 어떤 software가:

```text
JetPack 6.x required
```

라고 하면 실제로는:

```text
Ubuntu version
CUDA version
Driver version
L4T version
```

까지 같이 영향을 받을 수 있다.

---

# 13. CUDA란?

CUDA는 NVIDIA GPU에서 general-purpose parallel computing을 하기 위한
NVIDIA의 computing platform과 programming model이다.

쉽게 말하면:

> NVIDIA GPU를 그래픽 외의 계산에도 사용할 수 있게 하는 기술

이다.

예:

```text
CPU
 │
 │ CUDA API
 ▼
GPU
 │
 └── Parallel Computation
```

로봇에서는:

```text
Deep Learning
Point Cloud Processing
Image Processing
Matrix Operations
```

등에서 사용할 수 있다.

---

# 14. CUDA Core란?

NVIDIA GPU 내부에는 많은 연산 unit이 존재한다.

보통 설명할 때 CUDA Core라는 표현을 많이 사용한다.

CPU와 비교하면:

```text
CPU

적은 수의 강한 core

vs

GPU

매우 많은 병렬 연산 unit
```

이다.

하지만 CUDA Core 개수만 보고 GPU 전체 성능을 판단하면 안 된다.

성능은:

```text
Architecture
Clock
Memory Bandwidth
Tensor Core
Power Mode
Software
```

등에 영향을 받는다.

---

# 15. Tensor Core란?

NVIDIA GPU에는 AI 연산에 특화된 **Tensor Core**가 존재할 수 있다.

Tensor Core는 특히:

```text
Matrix Multiplication
Deep Learning
Mixed Precision
```

같은 연산을 빠르게 수행하도록 설계되었다.

예:

```text
Neural Network

Matrix × Matrix

        │
        ▼
   Tensor Core
```

그래서 Jetson은 AI inference workload에서 강하다.

---

# 16. cuDNN이란?

cuDNN은:

**CUDA Deep Neural Network library**

이다.

NVIDIA가 Deep Learning 연산을 GPU에서 효율적으로 수행할 수 있도록 제공하는 library다.

예:

```text
PyTorch
TensorFlow
    │
    ▼
  cuDNN
    │
    ▼
  CUDA
    │
    ▼
NVIDIA GPU
```

개발자가 모든 convolution이나 activation 연산을 직접 CUDA로 구현하지 않아도
optimized library를 사용할 수 있게 한다.

---

# 17. TensorRT란?

TensorRT는 NVIDIA의 **Deep Learning Inference Optimization Runtime**이다.

학습이 끝난 neural network를 실제 device에서 빠르게 추론할 수 있도록 최적화한다.

예:

```text
PyTorch Model
      │
      ▼
Optimization
      │
      ▼
TensorRT Engine
      │
      ▼
Jetson GPU
      │
      ▼
Fast Inference
```

TensorRT는 특히:

```text
Lower Precision
Layer Fusion
Kernel Optimization
Memory Optimization
```

등을 사용해 inference latency와 throughput을 개선할 수 있다.

---

# 18. Training과 Inference

Deep Learning에서:

```text
Training
Inference
```

는 다르다.

## Training

Model parameter를 학습.

```text
Dataset
   ↓
Neural Network
   ↓
Backpropagation
   ↓
Weight Update
```

보통 매우 많은 계산이 필요하다.

---

## Inference

이미 학습된 model을 사용해 prediction.

```text
Camera Image
   ↓
Trained Model
   ↓
Person detected
```

Jetson은 특히 Edge Inference에 많이 사용된다.

---

# 19. Jetson에서 PyTorch와 TensorRT

예를 들어 로봇 camera에서 object detection을 한다고 하자.

처음 개발:

```text
Python
  ↓
PyTorch
  ↓
GPU
```

성능 최적화 후:

```text
Model
  ↓
TensorRT
  ↓
Jetson GPU
```

구조로 가져갈 수 있다.

하지만 모든 model이 TensorRT로 자동 변환되는 것은 아니고
operator support와 version compatibility를 확인해야 한다.

---

# 20. Unified Memory Architecture

Jetson에서 중요한 특징 중 하나는 CPU와 GPU가 system memory를 공유하는 구조다.

일반 PC discrete GPU:

```text
CPU
 │
 └── System RAM

GPU
 │
 └── VRAM
```

CPU memory와 GPU memory가 물리적으로 분리되어 있다.

그래서 data transfer가 필요할 수 있다.

```text
System RAM
    │
    │ PCIe Copy
    ▼
GPU VRAM
```

---

Jetson은 integrated SoC 구조이므로:

```text
CPU
  \
   \
    Shared System Memory
   /
  /
GPU
```

형태로 memory를 공유한다.

---

# 21. Shared Memory가 무조건 Copy가 없는 것은 아니다

중요한 점:

> CPU와 GPU가 같은 physical memory를 사용할 수 있다고 해서
> 모든 software에서 자동으로 zero-copy가 되는 것은 아니다.

실제 data movement는:

```text
CUDA API
Memory allocation 방식
Framework
Driver
Buffer type
```

에 따라 달라진다.

그래서:

```text
Unified Architecture
≠
Always Zero Copy
```

이다.

---

# 22. 왜 Memory Bandwidth가 중요할까?

SLAM, Vision, AI는 큰 데이터를 계속 처리한다.

예:

```text
Camera Image
LiDAR Point Cloud
Tensor
Map
```

이 데이터는 CPU/GPU 사이에서 계속 읽고 써야 한다.

이때 중요한 성능 지표가:

```text
Memory Bandwidth
```

이다.

즉:

> Memory에서 얼마나 빠르게 데이터를 읽고 쓸 수 있는가

를 의미한다.

GPU가 아무리 빠르더라도 data 공급이 느리면 전체 성능이 제한될 수 있다.

---

# 23. Jetson의 Storage

Jetson model에 따라:

```text
eMMC
NVMe SSD
microSD
```

등을 사용할 수 있다.

로봇에서는:

```text
rosbag
LiDAR data
Video
Log
Map
```

같은 데이터가 매우 빠르게 쌓일 수 있다.

그래서 Storage capacity뿐 아니라:

```text
Read Speed
Write Speed
Endurance
```

도 중요하다.

---

# 24. Jetson에서 `nvidia-smi`

Desktop NVIDIA GPU에서:

```bash
nvidia-smi
```

는 매우 유명한 명령이다.

GPU usage, memory, process 등을 확인할 수 있다.

하지만 Jetson에서는 desktop discrete GPU 환경과 동일하게 동작하지 않거나
보여주는 정보가 제한될 수 있다.

Jetson에서는 대신:

```bash
tegrastats
```

를 매우 자주 사용한다.

---

# 25. `tegrastats`

`tegrastats`는 Jetson의 system resource를 모니터링하는 도구다.

실행:

```bash
tegrastats
```

다음과 같은 정보를 볼 수 있다.

```text
RAM usage
CPU usage
GPU usage
Temperature
Power-related values
Memory Controller load
```

예를 들어:

```text
GR3D_FREQ
```

같은 값을 볼 수 있는데,
GPU activity와 관련된 정보를 나타낸다.

---

# 26. `jtop`

Jetson에서는 `jtop`이라는 tool도 많이 사용한다.

보통 `jetson-stats` package를 통해 설치한다.

실행:

```bash
jtop
```

좀 더 보기 편한 UI로:

```text
CPU
GPU
Memory
Temperature
Power
JetPack
CUDA
```

정보 등을 볼 수 있다.

---

# 27. Jetson Version 확인

Jetson에서 환경을 확인할 때는 version 확인이 매우 중요하다.

예:

```bash
cat /etc/os-release
```

Ubuntu version 확인.

```bash
uname -a
```

Kernel 정보 확인.

```bash
uname -m
```

Architecture 확인.

Jetson에서는:

```text
aarch64
```

가 나오는 경우가 일반적이다.

---

# 28. L4T 확인

다음 파일을 확인하는 방법도 자주 사용된다.

```bash
cat /etc/nv_tegra_release
```

Jetson Linux / L4T 관련 release 정보를 볼 수 있다.

환경에 따라 파일이 없거나 다른 방식으로 확인해야 할 수도 있다.

---

# 29. CUDA Version 확인

예:

```bash
nvcc --version
```

CUDA compiler가 설치되어 있다면 version 확인 가능.

또는:

```bash
ls /usr/local/cuda
```

등을 확인하기도 한다.

중요한 점:

```text
CUDA Toolkit Version
Driver Capability
Framework Compatibility
```

를 서로 구분해야 한다.

---

# 30. Power Mode

Jetson은 Edge Device이기 때문에 전력 제한이 매우 중요하다.

Desktop GPU처럼 항상 최대 전력으로 동작하도록 설계된 것이 아니다.

Jetson에서는 여러 power mode가 존재할 수 있다.

예:

```text
Low Power Mode
Balanced Mode
Max Performance Mode
```

model마다 이름과 구성은 다를 수 있다.

Power mode는:

```text
CPU Core Count
CPU Clock
GPU Clock
Power Budget
```

등에 영향을 줄 수 있다.

---

# 31. `nvpmodel`

Jetson power mode를 관리할 때:

```bash
nvpmodel
```

을 사용한다.

현재 mode 확인:

```bash
sudo nvpmodel -q
```

정확한 option과 mode 번호는 Jetson model에 따라 다르므로
무작정 mode를 변경하지 않고 device documentation을 확인해야 한다.

---

# 32. Clock과 Performance

CPU/GPU는 항상 최대 clock으로 동작하지 않는다.

필요에 따라 clock이 변한다.

```text
Low workload
     ↓
Lower Clock

Heavy workload
     ↓
Higher Clock
```

전력과 열을 줄이기 위해서다.

Performance test를 할 때는 power mode와 clock 상태를 함께 확인해야 한다.

---

# 33. Thermal Throttling

Jetson이 너무 뜨거워지면 hardware 보호를 위해 성능을 낮출 수 있다.

이를:

```text
Thermal Throttling
```

이라고 한다.

예:

```text
GPU 100%
   ↓
Temperature 상승
   ↓
Thermal Limit
   ↓
Clock 감소
   ↓
Performance 감소
```

따라서 실험 중 FPS나 SLAM frequency가 떨어진다면
software 문제뿐 아니라 thermal 문제도 확인해야 한다.

---

# 34. Cooling이 중요한 이유

Jetson AGX급 module은 높은 compute performance를 낼 수 있지만
동시에 상당한 열이 발생한다.

그래서:

```text
Heatsink
Fan
Airflow
Thermal Interface
```

가 중요하다.

로봇 내부에 Jetson을 넣을 때 enclosure만 보고 설계하면 안 되고,
열이 빠져나갈 경로까지 고려해야 한다.

---

# 35. 로봇에서 Power Budget

로봇은 전체 배터리 에너지를 여러 장치가 나눠 사용한다.

```text
Battery
  │
  ├── Motors
  ├── Jetson
  ├── LiDAR
  ├── Camera
  ├── Network
  └── Other Sensors
```

Jetson compute performance를 높이면 power consumption과 heat도 증가할 수 있다.

그래서:

```text
Performance
Power
Thermal
Battery Life
```

사이에 trade-off가 존재한다.

---

# 36. Jetson에서 ROS 2 실행

Jetson에서는 일반 Ubuntu PC와 비슷하게 ROS 2를 사용할 수 있다.

예:

```bash
source /opt/ros/humble/setup.bash
```

```bash
ros2 node list
```

```bash
ros2 topic list
```

하지만 Jetson은 ARM64이므로 package와 binary가 ARM64를 지원해야 한다.

---

# 37. FAST-LIO2는 CPU일까 GPU일까?

FAST-LIO2 같은 SLAM을 보면:

```text
LiDAR
IMU
 │
 ▼
FAST-LIO2
 │
 ▼
Odometry / Map
```

기본적으로 상당 부분의 계산이 CPU 기반으로 실행될 수 있다.

즉:

> Jetson에 GPU가 있다고 모든 robot algorithm이 자동으로 GPU를 사용하는 것은 아니다.

어떤 algorithm이 GPU를 사용하는지는 구현에 따라 결정된다.

---

# 38. GPU 사용 여부는 코드가 결정한다

예:

```cpp
for (...) {
    process_point();
}
```

일반 C++ 코드라면 기본적으로 CPU에서 실행된다.

CUDA kernel을 사용하면:

```cpp
kernel<<<blocks, threads>>>();
```

GPU에서 실행할 수 있다.

PyTorch에서도:

```python
tensor = tensor.to("cuda")
```

등을 통해 GPU tensor로 옮겨 연산할 수 있다.

즉:

```text
Jetson has GPU
≠
Every program uses GPU
```

---

# 39. Vision60에서 Jetson 역할

Vision60 같은 로봇에서 computing architecture를 단순화하면:

```text
Sensors

LiDAR
IMU
Camera
Joint Encoder
      │
      ▼
┌────────────────────────┐
│ Xavier / Jetson Orin   │
│                        │
│ CPU                    │
│ ├── ROS 2              │
│ ├── Sensor Driver      │
│ ├── FAST-LIO2          │
│ └── Navigation         │
│                        │
│ GPU                    │
│ ├── Vision             │
│ ├── AI Inference       │
│ └── CUDA Processing    │
└────────────┬───────────┘
             │
             ▼
           MCU
             │
             ▼
          Actuator
```

Jetson은 high-level perception/computation을 담당하고,
MCU는 hardware에 가까운 motor control을 담당하는 구조로 이해할 수 있다.

---

# 40. Jetson과 MCU의 차이

| Jetson | MCU |
|---|---|
| Linux 실행 | Bare-metal / RTOS 가능 |
| ROS 2 실행 가능 | Low-level control |
| CPU + GPU | 작은 CPU |
| Perception / SLAM | Motor / Sensor timing |
| 높은 computing capability | 높은 deterministic control |

예:

```text
Jetson
"앞으로 이동해야 한다"

        ↓

MCU
"각 joint motor에 정확히 언제 얼마의 torque를 줄 것인가"
```

정도로 역할을 나누어 생각할 수 있다.

---

# 41. Jetson Software Compatibility

Jetson에서 설치 문제를 만났다면 다음을 확인한다.

```text
1. Jetson Model
2. CPU Architecture
3. Ubuntu Version
4. JetPack Version
5. L4T Version
6. CUDA Version
7. cuDNN Version
8. TensorRT Version
9. Python Version
10. Framework Version
```

특히:

```text
PyTorch
TensorFlow
CUDA
TensorRT
```

는 version compatibility가 매우 중요하다.

---

# 42. Docker를 사용할 때도 JetPack이 중요하다

Jetson에서 Docker를 사용할 때 image가 단순히:

```text
linux/arm64
```

를 지원한다고 끝이 아니다.

GPU를 사용하려면:

```text
Jetson Driver
CUDA Runtime
Container Runtime
JetPack
```

의 호환성이 맞아야 한다.

그래서 Jetson용 NVIDIA container image를 사용할 때
JetPack/L4T version을 함께 확인하는 경우가 많다.

---

# 43. Jetson Debugging Mental Model

Jetson에서 어떤 AI program이 안 돌아간다면 다음 순서로 생각한다.

```text
Application
   │
   ▼
Framework
PyTorch / TensorRT
   │
   ▼
CUDA / cuDNN
   │
   ▼
NVIDIA Driver
   │
   ▼
Jetson Linux
   │
   ▼
Hardware
GPU / Memory
```

예를 들어:

```text
PyTorch CUDA unavailable
```

라고 나왔다면 바로 GPU가 고장났다고 판단하면 안 된다.

```text
PyTorch build?
CUDA version?
Driver?
Container?
Environment variable?
```

등 여러 층을 확인해야 한다.

---

# 44. 실무 확인 명령어

## Architecture

```bash
uname -m
```

---

## OS

```bash
cat /etc/os-release
```

---

## Kernel

```bash
uname -a
```

---

## CPU

```bash
lscpu
```

---

## Memory

```bash
free -h
```

---

## Storage

```bash
df -h
```

---

## Jetson Resource

```bash
tegrastats
```

---

## Power Mode

```bash
sudo nvpmodel -q
```

---

## CUDA Compiler

```bash
nvcc --version
```

---

# 45. Mini Practice

Jetson에 SSH로 접속해서 다음을 확인한다.

```bash
uname -m
```

예상:

```text
aarch64
```

---

```bash
cat /etc/os-release
```

Ubuntu version을 확인한다.

---

```bash
lscpu
```

CPU 정보를 확인한다.

---

```bash
free -h
```

RAM 사용량 확인.

---

```bash
df -h
```

Storage 사용량 확인.

---

```bash
tegrastats
```

CPU/GPU/RAM/temperature 정보를 관찰한다.

ROS 2나 SLAM을 실행한 상태에서 `tegrastats` 값이 어떻게 변하는지 비교하면 더 좋다.

---

# 46. 오늘의 핵심

Jetson을 이해할 때 가장 중요한 구조는 다음과 같다.

```text
Jetson Hardware

ARM CPU
+
NVIDIA GPU
+
Shared Memory
+
Sensor I/O

        │
        ▼

Jetson Linux

        │
        ▼

Ubuntu

        │
        ▼

JetPack Libraries

CUDA
cuDNN
TensorRT

        │
        ▼

ROS 2 / SLAM / AI
```

그리고 다음을 반드시 기억한다.

```text
Jetson ≠ GPU

JetPack ≠ Ubuntu

CUDA ≠ GPU

TensorRT ≠ Training Framework

ARM64 support
≠
Jetson compatibility automatically guaranteed
```

---

# 47. 가장 중요한 질문

Jetson에서 어떤 software를 설치하거나 실행할 때 항상 다음을 먼저 확인한다.

```text
What Jetson model?

What JetPack version?

What Ubuntu version?

What architecture?

What CUDA version?

What library version?
```

이 정보를 모르면 Jetson software 문제를 정확히 디버깅하기 어렵다.

---

# Next Chapter

## Chapter 5. Hardware Interfaces — Ethernet, CAN, PCIe, USB

다음 Chapter에서는 Jetson에 센서와 장치가 실제로 어떻게 연결되는지 다룬다.

- Ethernet은 단순히 인터넷 연결용인가?
- MAC address와 IP address는 무엇이 다른가?
- USB sensor는 Linux에서 어떻게 보이는가?
- Serial 통신이란?
- CAN Bus는 왜 로봇과 자동차에서 많이 사용하는가?
- PCIe는 무엇인가?
- M.2는 PCIe와 같은 것인가?
- LiDAR는 Ethernet으로 어떻게 데이터를 보내는가?
- Camera는 USB / CSI 중 무엇을 사용할까?
- Xavier와 Orin 사이를 Ethernet으로 연결한다는 것은 무슨 뜻인가?
- Robot 내부 network와 Internet은 어떻게 다른가?

Chapter 5에서는 이제:

```text
Sensor
   │
   ▼
Physical Interface
   │
   ▼
Linux Driver
   │
   ▼
ROS 2
```

가 실제로 어떻게 연결되는지 살펴본다.