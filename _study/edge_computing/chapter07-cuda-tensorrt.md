---
title: "Chapter 7. CUDA & TensorRT for Robotics"
importance: 8
---

> **Goal:** Jetson의 NVIDIA GPU를 프로그램이 실제로 어떻게 사용하는지 이해한다.
> CPU와 GPU의 차이, CUDA kernel, thread/block/grid, GPU memory, PyTorch CUDA,
> TensorRT, FP16/INT8, latency/throughput을 로봇 실무 관점에서 연결한다.

---

# 1. GPU는 있다고 자동으로 쓰는 것이 아니다

Jetson에는 NVIDIA GPU가 들어 있다.

하지만 Jetson에서 프로그램을 실행한다고 해서
자동으로 GPU를 사용하는 것은 아니다.

예를 들어 일반적인 C++ 코드를 작성하면:

```cpp
for (int i = 0; i < 1000000; i++) {
    output[i] = input[i] * 2;
}
```

기본적으로 이 코드는 CPU에서 실행된다.

```text
C++ Code
   │
   ▼
CPU
```

Jetson 안에 GPU가 존재하더라도,
프로그램이 GPU 연산을 요청하지 않으면 GPU는 그 계산에 사용되지 않는다.

즉:

```text
Jetson has GPU
≠
Every program uses GPU
```

이다.

---

# 2. CPU와 GPU의 차이 다시 보기

CPU는 복잡한 제어 흐름과 다양한 종류의 작업에 강하다.

```text
CPU

Core 0
Core 1
Core 2
Core 3
...
```

GPU는 같은 종류의 계산을 매우 많이 병렬로 수행하는 데 강하다.

```text
GPU

Thread Thread Thread Thread
Thread Thread Thread Thread
Thread Thread Thread Thread
...
```

예를 들어 1,000,000개의 숫자에 같은 계산을 적용한다면 GPU가 유리할 수 있다.

---

# 3. Parallel Computing

병렬 계산은 여러 계산을 동시에 수행하는 것이다.

CPU 방식:

```text
1
2
3
4
5
6
...
```

순서대로 처리하는 경우가 많다.

GPU 방식은 독립적인 계산이라면:

```text
1 2 3 4 5 6 ...
↓ ↓ ↓ ↓ ↓ ↓
동시에 많은 작업 실행
```

이 가능하다.

물론 실제 hardware scheduling은 훨씬 복잡하지만,
개념적으로는 이 차이를 이해하면 된다.

---

# 4. 어떤 연산이 GPU에 적합할까?

GPU에 적합한 연산은 보통:

```text
같은 계산을
많은 데이터에
반복적으로 적용
```

하는 형태다.

예:

```text
Matrix Multiplication
Image Processing
Neural Network
Point-wise Operation
Large Vector Operation
```

---

# 5. 어떤 연산은 GPU에 잘 맞지 않을까?

다음처럼:

```text
복잡한 조건문
불규칙한 memory access
작은 데이터
순차적인 dependency가 강한 계산
```

은 GPU의 장점을 충분히 활용하지 못할 수 있다.

예:

```cpp
if (condition_a) {
    ...
} else if (condition_b) {
    ...
} else {
    ...
}
```

가 매우 복잡하게 얽혀 있는 알고리즘은 GPU로 옮겼다고 무조건 빨라지지 않는다.

---

# 6. CUDA란?

CUDA는 NVIDIA가 만든:

> NVIDIA GPU를 general-purpose computing에 사용하기 위한 programming platform

이다.

CUDA를 사용하면 CPU가 GPU에게 계산을 요청할 수 있다.

```text
CPU Program
    │
    │ Launch
    ▼
CUDA Kernel
    │
    ▼
GPU
```

---

# 7. Host와 Device

CUDA에서는 자주 다음 용어를 사용한다.

```text
Host
Device
```

보통:

```text
Host = CPU side
Device = GPU side
```

를 의미한다.

예:

```text
Host Code
   │
   ▼
CPU

Device Code
   │
   ▼
GPU
```

---

# 8. CUDA Kernel

GPU에서 실행되는 함수를 **Kernel**이라고 한다.

예를 들어 매우 단순화한 CUDA kernel:

```cpp
__global__ void multiplyByTwo(float* data)
{
    int i = threadIdx.x;
    data[i] *= 2.0f;
}
```

CPU에서 kernel을 실행할 때:

```cpp
multiplyByTwo<<<1, 256>>>(data);
```

같은 문법을 사용할 수 있다.

여기서:

```text
<<< >>>
```

안에는 GPU에서 얼마나 많은 thread를 실행할지에 대한 configuration이 들어간다.

---

# 9. CUDA Thread

GPU에서는 작은 작업 단위를 thread라고 한다.

예:

```text
Data

0 1 2 3 4 5 6 7

각 데이터마다

Thread 0
Thread 1
Thread 2
Thread 3
...
```

를 대응시킬 수 있다.

예:

```text
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
...
```

이렇게 동일한 계산을 병렬로 수행한다.

---

# 10. Thread, Block, Grid

CUDA에서는 thread를 계층적으로 구성한다.

```text
Grid
 │
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 │
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 │
 └── ...
```

구조는:

```text
Grid
 ↓
Block
 ↓
Thread
```

이다.

---

# 11. 왜 Block이 필요할까?

GPU에는 실제로 수많은 thread가 실행될 수 있다.

모든 thread를 하나의 거대한 단위로 관리하지 않고
block 단위로 묶어서 scheduling과 memory sharing을 효율적으로 관리한다.

```text
Grid

Block 0
Block 1
Block 2
Block 3
...
```

각 block 안에는 여러 thread가 있다.

---

# 12. Thread Index

각 thread는 자신의 위치를 알 수 있다.

CUDA에서는:

```cpp
threadIdx.x
blockIdx.x
blockDim.x
```

같은 값을 사용한다.

전체 데이터 index를 계산할 때 흔히:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

형태를 사용한다.

즉:

```text
어느 Block인가?
+
Block 안에서 몇 번째 Thread인가?
=
전체 데이터 index
```

이다.

---

# 13. 예제: Vector Addition

두 vector를 더한다고 하자.

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```

결과:

```text
C = [6, 8, 10, 12]
```

CPU에서는:

```cpp
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

GPU에서는 conceptually:

```text
Thread 0 → C[0] = A[0] + B[0]
Thread 1 → C[1] = A[1] + B[1]
Thread 2 → C[2] = A[2] + B[2]
Thread 3 → C[3] = A[3] + B[3]
```

를 병렬로 수행할 수 있다.

---

# 14. GPU에는 정말 수천 개의 Thread가 동시에 실행될까?

CUDA program에서는 수천~수백만 개의 thread를 생성할 수 있다.

하지만 생성한 모든 thread가 물리적으로 정확히 동시에 실행되는 것은 아니다.

GPU hardware가 thread들을:

```text
Schedule
Execute
Pause
Resume
```

하면서 처리한다.

중요한 것은 개발자가 매우 많은 독립적인 작은 작업을 정의하고,
GPU scheduler가 이를 병렬 hardware에 배치한다는 것이다.

---

# 15. Warp

NVIDIA GPU에서는 thread가 보통 **Warp**라는 단위로 묶여 실행된다.

일반적으로 한 warp는:

```text
32 threads
```

로 구성된다.

```text
Warp

Thread 0
Thread 1
...
Thread 31
```

이 thread들이 비슷한 instruction을 함께 수행한다.

---

# 16. Branch Divergence

같은 warp 안의 thread들이 서로 다른 조건문을 실행하면 성능이 떨어질 수 있다.

예:

```cpp
if (idx % 2 == 0) {
    pathA();
} else {
    pathB();
}
```

한 warp 안에서:

```text
Thread 0 → A
Thread 1 → B
Thread 2 → A
Thread 3 → B
```

처럼 갈라지면 GPU가 두 path를 효율적으로 완전히 동시에 처리하지 못할 수 있다.

이를:

```text
Branch Divergence
```

라고 한다.

---

# 17. GPU Memory

GPU 계산에서는 memory도 중요하다.

Desktop discrete GPU에서는 보통:

```text
CPU
 │
System RAM
 │
PCIe
 │
GPU
 │
VRAM
```

구조다.

GPU가 계산하려면 데이터를 GPU memory로 옮겨야 하는 경우가 많다.

---

# 18. CUDA Memory Copy

CUDA에서는 이런 개념을 자주 본다.

```text
CPU RAM
   │
   │ cudaMemcpy
   ▼
GPU Memory
```

계산 후:

```text
GPU Memory
   │
   │ cudaMemcpy
   ▼
CPU RAM
```

으로 다시 가져올 수도 있다.

GPU 연산 자체가 빨라도 copy 시간이 길면 전체 성능이 나빠질 수 있다.

---

# 19. Data Transfer Overhead

예를 들어:

```text
CPU → GPU copy: 5 ms
GPU calculation: 1 ms
GPU → CPU copy: 5 ms
```

라면 실제 전체 시간은:

```text
11 ms
```

이다.

GPU 연산만 보면 1 ms지만,
전체 pipeline은 그렇지 않다.

그래서:

> GPU optimization에서 compute 시간뿐 아니라 data movement가 매우 중요하다.

---

# 20. Jetson에서는 Memory가 조금 다르다

Chapter 4에서 배운 것처럼 Jetson은 integrated SoC 구조다.

```text
CPU
  \
   \
    System Memory
   /
  /
GPU
```

CPU와 GPU가 같은 physical system memory를 공유할 수 있다.

그래서 discrete GPU system보다 data movement를 효율적으로 구성할 가능성이 있다.

---

# 21. 하지만 Shared Memory = Zero Copy는 아니다

다시 강조하면:

```text
Shared Physical Memory
≠
Automatically Zero Copy
```

software가 memory를 어떻게 allocate하고
framework가 buffer를 어떻게 관리하는지에 따라 copy가 발생할 수 있다.

예:

```text
Camera Buffer
   ↓
CPU Buffer
   ↓
Framework Copy
   ↓
GPU Tensor
```

처럼 불필요한 copy가 생길 수도 있다.

---

# 22. Zero-Copy

Zero-copy는 불필요한 data copy를 줄이고
같은 memory buffer를 여러 component가 공유하도록 하는 최적화 개념이다.

예:

```text
Camera
  │
  ▼
Shared Buffer
  ├── CPU
  └── GPU
```

로봇 vision pipeline에서는 매우 중요할 수 있다.

---

# 23. Pinned Memory

CUDA에서:

```text
Pinned Memory
```

라는 개념도 등장한다.

일반적인 pageable host memory와 달리
OS가 memory를 swap out하지 않도록 고정된 memory다.

CPU ↔ GPU transfer를 빠르게 만드는 데 사용될 수 있다.

다만 pinned memory를 너무 많이 사용하면 system memory 관리에 부담이 될 수 있다.

---

# 24. CUDA Stream

GPU에서는 operation을 비동기적으로 실행할 수 있다.

CUDA Stream은 GPU operation의 실행 순서를 관리하는 queue처럼 생각할 수 있다.

```text
Stream 1
H2D Copy
Kernel
D2H Copy
```

다른 stream:

```text
Stream 2
H2D Copy
Kernel
D2H Copy
```

를 사용하면 일부 작업을 겹쳐 실행할 수 있다.

---

# 25. Synchronous vs Asynchronous

동기 방식:

```text
CPU
 ↓
GPU 작업 시작
 ↓
GPU 끝날 때까지 기다림
 ↓
다음 작업
```

비동기 방식:

```text
CPU
 ↓
GPU 작업 요청
 ↓
CPU 다른 작업 수행
```

GPU pipeline에서는 asynchronous programming이 매우 중요할 수 있다.

---

# 26. PyTorch에서 GPU 사용

PyTorch에서는 CUDA 사용이 더 쉽게 추상화되어 있다.

예:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
```

기본적으로 CPU tensor일 수 있다.

GPU로 이동:

```python
x = x.to("cuda")
```

그러면 GPU에서 사용할 수 있는 tensor가 된다.

---

# 27. `.to("cuda")`가 의미하는 것

개념적으로:

```text
CPU Tensor
    │
    │ .to("cuda")
    ▼
GPU Tensor
```

이다.

Discrete GPU에서는 실제 memory copy가 일어날 수 있다.

Jetson에서는 shared memory architecture의 영향을 받지만,
framework 관점에서는 CPU tensor와 CUDA tensor를 여전히 구분한다.

---

# 28. Tensor의 Device 확인

PyTorch에서는:

```python
print(x.device)
```

를 사용할 수 있다.

예:

```text
cpu
```

또는:

```text
cuda:0
```

가 나올 수 있다.

---

# 29. Model도 GPU로 보내야 한다

Tensor만 GPU에 있고 model이 CPU에 있으면 계산할 수 없다.

예:

```python
model = model.to("cuda")
input_tensor = input_tensor.to("cuda")
```

구조:

```text
Model
  │
  ▼
GPU

Input
  │
  ▼
GPU

      ↓

Inference
```

---

# 30. CPU/GPU Device Mismatch

PyTorch에서 자주 보는 error가:

```text
Expected all tensors to be on the same device
```

같은 형태다.

예:

```text
Model → cuda
Input → cpu
```

이면 연산이 불가능할 수 있다.

따라서:

```text
Model Device
=
Input Device
```

를 맞춰야 한다.

---

# 31. Neural Network Inference

학습된 model로 prediction하는 과정이 inference다.

예:

```text
Camera
  ↓
Image Tensor
  ↓
GPU
  ↓
Neural Network
  ↓
Bounding Boxes
```

Jetson에서는 이러한 inference를 local edge에서 처리할 수 있다.

---

# 32. TensorRT

TensorRT는 NVIDIA가 제공하는
**Deep Learning Inference Optimization Runtime**이다.

목표는:

```text
Trained Model
    │
    ▼
Optimization
    │
    ▼
Efficient GPU Execution
```

이다.

---

# 33. PyTorch와 TensorRT 관계

개념적으로:

```text
PyTorch
   │
   │ Model Export / Conversion
   ▼
ONNX or other representation
   │
   ▼
TensorRT
   │
   ▼
TensorRT Engine
   │
   ▼
Jetson GPU
```

같은 pipeline을 사용할 수 있다.

실제 conversion 경로는 model과 toolchain에 따라 달라진다.

---

# 34. TensorRT Engine

TensorRT는 model을 분석하여 optimized execution engine을 생성할 수 있다.

```text
Neural Network

Conv
ReLU
Conv
BN
ReLU
...

      │
      ▼
TensorRT Optimization
      │
      ▼
Optimized Engine
```

이 engine은 특정 hardware/software 환경과 밀접하게 연관될 수 있다.

---

# 35. Layer Fusion

TensorRT optimization 중 하나로 자주 언급되는 것이 Layer Fusion이다.

예:

```text
Convolution
   ↓
BatchNorm
   ↓
Activation
```

을 각각 따로 실행하는 대신
가능한 경우 더 적은 GPU operation으로 합쳐 실행할 수 있다.

```text
Conv + BN + Activation
        ↓
   Fused Operation
```

이를 통해:

```text
Kernel Launch Overhead 감소
Memory Access 감소
Latency 감소
```

효과를 기대할 수 있다.

---

# 36. Precision

AI 계산에서는 숫자를 어떤 precision으로 표현할지도 중요하다.

대표적으로:

```text
FP32
FP16
INT8
```

을 많이 본다.

---

# 37. FP32

FP32는:

```text
32-bit Floating Point
```

이다.

일반적인 deep learning 계산의 기준 precision 중 하나다.

장점:

```text
높은 numerical precision
```

단점:

```text
Memory 사용량 큼
연산량 큼
```

---

# 38. FP16

FP16은:

```text
16-bit Floating Point
```

이다.

FP32보다 적은 memory를 사용하고
지원되는 GPU에서는 더 빠른 연산이 가능할 수 있다.

```text
FP32
32 bit

FP16
16 bit
```

따라서:

```text
Memory 감소
Bandwidth 절약
Tensor Core 활용
```

등의 장점이 있을 수 있다.

---

# 39. INT8

INT8은:

```text
8-bit Integer
```

이다.

더 낮은 precision을 사용함으로써
더 높은 inference performance와 낮은 memory 사용량을 얻을 수 있다.

하지만 model accuracy가 영향을 받을 수 있다.

---

# 40. Quantization

Floating point model을 INT8 같은 낮은 precision으로 변환하는 과정을:

```text
Quantization
```

이라고 한다.

```text
FP32 Model
    │
    ▼
Quantization
    │
    ▼
INT8 Model
```

정확한 scale을 결정하기 위해 calibration data가 필요한 방식도 있다.

---

# 41. Precision Trade-off

일반적인 관계를 단순화하면:

```text
FP32

높은 precision
높은 memory
낮은 speed 가능성

        ↓

FP16

        ↓

INT8

낮은 precision
낮은 memory
높은 speed 가능성
```

하지만 항상 INT8이 무조건 더 좋은 것은 아니다.

Model accuracy와 hardware support를 함께 봐야 한다.

---

# 42. Latency

Latency는 하나의 입력을 처리하는 데 걸리는 시간이다.

예:

```text
Camera Frame
    │
    ▼
Model
    │
    ▼
Detection

Total = 20 ms
```

이때 inference latency는:

```text
20 ms
```

이다.

로봇에서는 latency가 매우 중요하다.

---

# 43. Throughput

Throughput은 일정 시간 동안 몇 개의 input을 처리할 수 있는지를 의미한다.

예:

```text
100 images / second
```

라면 throughput은 대략:

```text
100 FPS
```

라고 볼 수 있다.

---

# 44. Latency와 Throughput은 다르다

예를 들어 batching을 크게 하면:

```text
Batch = 32
```

GPU utilization이 좋아져 throughput은 증가할 수 있다.

하지만 한 input이 결과를 받기까지 기다리는 시간은 길어질 수 있다.

즉:

```text
High Throughput
≠
Low Latency
```

이다.

---

# 45. Robot에서는 Latency가 더 중요한 경우가 많다

Cloud AI에서는 throughput이 매우 중요할 수 있다.

예:

```text
수천 개 image 처리
```

하지만 로봇에서는:

```text
Camera
  ↓
Obstacle Detection
  ↓
Decision
  ↓
Control
```

까지 빠르게 이어져야 한다.

그래서 real-time robotics에서는:

```text
Low Latency
```

가 특히 중요하다.

---

# 46. FPS

Camera나 inference에서:

```text
FPS
```

를 많이 사용한다.

FPS:

```text
Frames Per Second
```

예:

```text
30 FPS
```

이면 초당 30 frame 처리라는 뜻이다.

하지만:

```text
30 FPS
```

라고 해서 latency가 반드시:

```text
33 ms
```

인 것은 아니다.

Pipeline과 batching 방식에 따라 다를 수 있다.

---

# 47. GPU Utilization

GPU가 얼마나 바쁘게 동작하는지 확인하는 값이다.

Jetson에서는:

```bash
tegrastats
```

등을 통해 GPU 관련 utilization을 확인할 수 있다.

예:

```text
GR3D_FREQ
```

와 같은 항목을 볼 수 있다.

---

# 48. GPU 사용률이 100%면 좋은가?

항상 그렇지는 않다.

GPU 100%는 hardware를 잘 활용하고 있다는 의미일 수도 있지만,
전체 system이 bottleneck에 걸려 있다는 뜻일 수도 있다.

확인해야 하는 것:

```text
Latency
Temperature
Power
Memory
CPU usage
Input queue
Dropped frame
```

등이다.

---

# 49. Bottleneck

전체 pipeline에서 성능을 제한하는 부분을 bottleneck이라고 한다.

예:

```text
Camera
  │
  ▼
CPU Preprocessing   ← 40 ms
  │
  ▼
GPU Inference       ← 5 ms
  │
  ▼
Postprocessing      ← 10 ms
```

GPU inference만 최적화해도 전체 latency는 크게 줄지 않을 수 있다.

실제 bottleneck은 CPU preprocessing이기 때문이다.

---

# 50. Amdahl's Law 관점

프로그램의 일부만 빨라지면
전체 성능 향상에는 한계가 있다.

예:

```text
Total 100 ms

CPU part = 90 ms
GPU-able part = 10 ms
```

GPU 부분을 10배 빠르게 만들어도:

```text
90 ms
+
1 ms
=
91 ms
```

이다.

즉 전체가 10배 빨라지는 것이 아니다.

---

# 51. SLAM을 GPU로 옮기면 무조건 빨라질까?

아니다.

FAST-LIO2 같은 SLAM에는:

```text
Point Cloud Processing
Nearest Neighbor Search
Kalman Filter
Matrix Operations
Map Update
Conditional Logic
```

등 다양한 계산이 섞여 있다.

일부는 GPU에 적합할 수 있지만,
일부는 CPU에서 효율적일 수 있다.

따라서:

```text
SLAM
   │
   ├── GPU-friendly operation
   └── CPU-friendly operation
```

을 분석해야 한다.

---

# 52. GPU Porting Cost

CPU algorithm을 GPU로 옮기면 다음 문제가 생긴다.

```text
CUDA implementation 필요
Memory management
Synchronization
Debugging
Data transfer
Build complexity
Hardware dependency
```

즉 단순히:

```text
CPU slow
→ GPU로 이동
```

은 아니다.

---

# 53. Point Cloud와 GPU

Point cloud는 많은 point에 비슷한 연산을 수행하는 부분이 있어
GPU에 적합한 작업도 존재한다.

예:

```text
Point Transform
Filtering
Voxelization
Feature Extraction
```

하지만 tree search나 복잡한 data structure는
구현 방식에 따라 GPU 최적화 난이도가 높을 수 있다.

---

# 54. Vision에서 GPU가 특히 강한 이유

Camera image는 매우 규칙적인 grid 형태의 data다.

```text
1920 × 1080 pixels
```

각 pixel이나 tensor element에 비슷한 연산을 반복한다.

따라서:

```text
Convolution
Matrix Multiplication
Image Resize
Normalization
```

등은 GPU parallelism과 잘 맞는다.

---

# 55. Vision60에서 GPU 활용 예

Vision60의 Jetson에서 GPU를 활용할 수 있는 예:

```text
Camera
  │
  ▼
Object Detection
  │
  ▼
Obstacle / Person Detection

Camera
  │
  ▼
Semantic Segmentation
  │
  ▼
Terrain Understanding

LiDAR + Camera
  │
  ▼
Sensor Fusion Network
```

등이 있다.

---

# 56. CPU와 GPU 역할 분담

Vision60에서 단순화하면:

```text
             Jetson

┌─────────────────────────────┐
│                             │
│ CPU                         │
│ ├── Linux                   │
│ ├── ROS 2                   │
│ ├── Sensor Driver           │
│ ├── FAST-LIO2               │
│ └── Navigation / Control    │
│                             │
│ GPU                         │
│ ├── Deep Learning           │
│ ├── Vision                  │
│ ├── Tensor Operations       │
│ └── CUDA Parallel Compute   │
│                             │
└─────────────────────────────┘
```

실제로는 application마다 CPU/GPU 경계가 달라진다.

---

# 57. CPU-GPU Pipeline

예를 들어 camera AI pipeline:

```text
Camera
  │
  ▼
Capture
CPU / hardware
  │
  ▼
Preprocessing
CPU or GPU
  │
  ▼
Inference
GPU
  │
  ▼
Postprocessing
CPU or GPU
  │
  ▼
ROS 2 Publisher
CPU
```

성능 최적화에서는 전체 pipeline을 봐야 한다.

---

# 58. TensorRT와 ROS 2

TensorRT inference를 ROS 2 node 안에서 실행할 수도 있다.

```text
Camera Driver
     │
     │ ROS 2 Image
     ▼
Detection Node
     │
     ├── TensorRT
     │      │
     │      ▼
     │     GPU
     │
     ▼
Detection Topic
```

즉:

```text
ROS 2
→ communication/framework

TensorRT
→ inference runtime

CUDA
→ GPU computing platform
```

로 역할이 다르다.

---

# 59. CUDA와 TensorRT는 같은가?

아니다.

```text
CUDA

GPU general-purpose computing platform
```

반면:

```text
TensorRT

Deep Learning inference optimization/runtime
```

이다.

TensorRT 내부에서는 CUDA를 활용한다.

개념:

```text
TensorRT
   │
   ▼
CUDA
   │
   ▼
NVIDIA GPU
```

---

# 60. cuDNN과 TensorRT 차이

cuDNN:

```text
Deep learning primitive library
```

예:

```text
Convolution
Activation
Normalization
```

같은 operation을 optimized implementation으로 제공한다.

TensorRT:

```text
전체 inference graph를 분석하고 최적화
```

한다.

즉:

```text
Framework
   │
   ├── cuDNN
   │     └── optimized operations
   │
   └── TensorRT
         └── optimized inference engine
```

정도로 이해할 수 있다.

---

# 61. CUDA Synchronization

CPU와 GPU가 비동기적으로 동작하기 때문에
특정 순간에는 GPU 작업 완료를 기다려야 할 수 있다.

예:

```cpp
cudaDeviceSynchronize();
```

개념적으로:

```text
CPU
 │
 │ GPU task launch
 ▼
GPU running
 │
 │ wait
 ▼
cudaDeviceSynchronize
 │
 ▼
Continue
```

과도한 synchronization은 performance를 떨어뜨릴 수 있다.

---

# 62. Kernel Launch Overhead

GPU kernel을 실행하는 데도 일정 overhead가 있다.

매우 작은 연산을:

```text
GPU Kernel
GPU Kernel
GPU Kernel
GPU Kernel
...
```

수천 번 따로 실행하면 launch overhead가 커질 수 있다.

그래서 kernel fusion 같은 최적화가 중요하다.

---

# 63. Memory Access가 Compute보다 중요할 수도 있다

GPU 연산에서:

```text
Compute
```

보다:

```text
Memory Access
```

가 bottleneck이 되는 경우가 많다.

예:

```text
GPU Core는 계산 준비 완료
       │
       ▼
Memory에서 data 기다림
```

이 상황에서는 core 수를 늘려도 performance가 크게 좋아지지 않는다.

---

# 64. Memory Coalescing

GPU thread들이 연속적인 memory 위치를 효율적으로 읽도록 구성하면
memory bandwidth를 더 잘 활용할 수 있다.

예:

```text
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
Thread 3 → data[3]
```

같은 pattern이 효율적일 수 있다.

반대로 매우 불규칙한 access:

```text
Thread 0 → data[10000]
Thread 1 → data[4]
Thread 2 → data[820]
```

는 비효율적일 수 있다.

---

# 65. GPU Optimization 순서

GPU optimization을 할 때 무작정 kernel부터 수정하지 않는다.

먼저:

```text
1. Profile
2. Find bottleneck
3. Check data transfer
4. Check GPU utilization
5. Check memory bandwidth
6. Optimize algorithm
7. Re-profile
```

순서가 중요하다.

---

# 66. Profiling

Profiling은 프로그램이 실제로 어디에서 시간을 사용하는지 측정하는 것이다.

예:

```text
Preprocess      15 ms
Inference        8 ms
Postprocess      4 ms
ROS publish      1 ms
```

이런 정보를 알아야 무엇을 최적화해야 하는지 결정할 수 있다.

NVIDIA는 CUDA/Nsight 계열 profiling tool을 제공한다.

---

# 67. Optimization 전에 측정

중요한 원칙:

```text
Don't guess.
Measure.
```

예를 들어:

> "GPU inference가 느린 것 같다."

라고 생각했는데 profiling 결과:

```text
Inference = 5 ms
Image copy = 30 ms
```

라면 실제 문제는 inference가 아니다.

---

# 68. Real-Time과 Fast는 다르다

로봇에서:

```text
Fast
```

와:

```text
Real-Time
```

은 같은 의미가 아니다.

Fast:

```text
평균적으로 빠름
```

Real-Time:

```text
정해진 시간 안에 결과가 나오는 것이 중요
```

이다.

예:

```text
Average latency = 5 ms
```

여도 가끔:

```text
100 ms
```

까지 튄다면 control system에는 문제가 될 수 있다.

---

# 69. Jitter

Latency가 일정하지 않고 흔들리는 것을:

```text
Jitter
```

라고 한다.

예:

```text
5 ms
6 ms
5 ms
40 ms
6 ms
```

로봇에서는 평균 latency뿐 아니라 jitter도 중요하다.

---

# 70. GPU와 Determinism

GPU는 높은 throughput에는 강하지만
복잡한 scheduling과 resource sharing 때문에
hard real-time control에 직접 사용하는 데는 주의가 필요하다.

그래서 보통:

```text
GPU
→ Perception / AI

MCU / RT Controller
→ Hard real-time motor control
```

같은 역할 분리가 많이 사용된다.

---

# 71. Edge AI의 장점

Jetson에서 inference를 직접 수행하면:

```text
Camera
  ↓
Jetson
  ↓
AI
  ↓
Robot Decision
```

이 가능하다.

Cloud로 보내는 방식:

```text
Camera
  ↓
Internet
  ↓
Cloud
  ↓
AI
  ↓
Internet
  ↓
Robot
```

보다 latency와 network dependency를 줄일 수 있다.

---

# 72. Edge AI의 단점

하지만 Edge device에는 제한이 있다.

```text
Power Limit
Thermal Limit
Memory Limit
Compute Limit
Storage Limit
```

그래서 model optimization이 중요하다.

---

# 73. Model Optimization

Jetson에서는 다음 최적화를 고려할 수 있다.

```text
Smaller Model
FP16
INT8
TensorRT
Batch Size tuning
Input Resolution tuning
Pipeline optimization
```

즉 단순히 가장 큰 model을 올리는 것이 항상 최선은 아니다.

---

# 74. Accuracy vs Performance

예:

```text
Large Model
Accuracy ↑
Latency ↑
Power ↑

Small Model
Accuracy ↓ 가능
Latency ↓
Power ↓
```

로봇에서는 application requirement에 맞춰 trade-off를 정해야 한다.

---

# 75. Vision60 예제

예를 들어 Vision60이 공사 현장에서 사람을 탐지해야 한다고 하자.

```text
Camera
   │
   ▼
Jetson Orin
   │
   ▼
TensorRT Object Detector
   │
   ▼
Person Detection
   │
   ▼
ROS 2 Topic
   │
   ▼
Navigation
   │
   ▼
Stop / Avoid
```

여기에는:

```text
Chapter 1 → CPU/GPU/RAM
Chapter 2 → ARM64
Chapter 3 → Linux
Chapter 4 → Jetson/CUDA
Chapter 5 → Camera Interface
Chapter 6 → ROS 2
Chapter 7 → GPU/TensorRT
```

가 모두 연결되어 있다.

---

# 76. FAST-LIO2 + AI 구조

Vision60에서는 예를 들어:

```text
LiDAR + IMU
     │
     ▼
FAST-LIO2
     │
     ▼
Robot Pose
```

와:

```text
Camera
   │
   ▼
GPU AI
   │
   ▼
Object / Terrain Information
```

를 동시에 사용할 수 있다.

전체:

```text
                   Jetson

LiDAR ───────┐
IMU ─────────┤
             ▼
          FAST-LIO2
             │
             ▼
            Pose
             │
             ├──────────┐
             │          │
Camera ──────┼──────► GPU AI
             │          │
             │          ▼
             │      Perception
             │          │
             └────┬─────┘
                  ▼
              Navigation
```

---

# 77. GPU 사용 여부 확인

PyTorch:

```python
import torch

print(torch.cuda.is_available())
```

예:

```text
True
```

이면 PyTorch가 CUDA device를 사용할 수 있다는 뜻이다.

GPU 개수:

```python
print(torch.cuda.device_count())
```

Device 이름:

```python
print(torch.cuda.get_device_name(0))
```

---

# 78. CUDA Version 확인

Jetson에서:

```bash
nvcc --version
```

CUDA compiler toolkit version을 확인할 수 있다.

하지만:

```text
nvcc version
```

과:

```text
GPU driver capability
```

는 완전히 같은 개념은 아니다.

---

# 79. Jetson Monitoring

실제 inference 실행 중:

```bash
tegrastats
```

를 실행한다.

확인:

```text
CPU
GPU
RAM
Temperature
Power
```

GPU AI를 켰을 때와 껐을 때를 비교해 보면 좋다.

---

# 80. Mini Practice 1: CUDA 확인

Jetson에서:

```bash
uname -m
```

```bash
nvcc --version
```

```bash
tegrastats
```

를 실행한다.

---

# 81. Mini Practice 2: PyTorch GPU

Python:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

를 실행한다.

---

# 82. Mini Practice 3: CPU vs GPU Tensor

```python
import torch

x = torch.randn(1000, 1000)

print(x.device)

if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)
```

출력 예:

```text
cpu
cuda:0
```

---

# 83. Mini Practice 4: 실제 Bottleneck 생각하기

다음 pipeline이 있다고 하자.

```text
Camera Capture   5 ms
Preprocessing   20 ms
Inference        6 ms
Postprocessing   4 ms
ROS Publish      1 ms
```

질문:

```text
GPU inference를 6 ms → 3 ms로 줄이는 것이
가장 효과적인 최적화인가?
```

아니다.

현재 가장 큰 bottleneck은:

```text
Preprocessing = 20 ms
```

이다.

---

# 84. 오늘의 핵심

GPU programming의 핵심은 단순히:

```text
GPU = 빠르다
```

가 아니다.

더 정확하게는:

```text
GPU
=
많은 독립적 계산을
병렬로 처리하는 데 강한 processor
```

이다.

그리고 실제 performance는:

```text
Parallelism
Memory Access
Data Transfer
Kernel Launch
Synchronization
Precision
Thermal
Power
```

등에 모두 영향을 받는다.

---

# 85. 반드시 구분할 것

```text
GPU ≠ CUDA

CUDA ≠ TensorRT

TensorRT ≠ PyTorch

FP16 ≠ INT8

Latency ≠ Throughput

FPS ≠ Latency

Shared Memory ≠ Automatic Zero Copy

GPU Utilization 100%
≠
System is optimal

Fast
≠
Real-Time
```

---

# 86. Chapter 1~7 전체 Stack

지금까지의 모든 Chapter를 하나로 연결하면:

```text
┌──────────────────────────────────┐
│ Robot Application                │
│ SLAM / AI / Navigation           │
├──────────────────────────────────┤
│ ROS 2                            │
│ Node / Topic / DDS               │
├──────────────────────────────────┤
│ CUDA / TensorRT                  │
│ GPU Parallel Computing           │
├──────────────────────────────────┤
│ Linux / JetPack                  │
├──────────────────────────────────┤
│ ARM64                            │
├──────────────────────────────────┤
│ CPU / GPU / RAM / Storage        │
├──────────────────────────────────┤
│ Ethernet / CAN / USB / PCIe      │
├──────────────────────────────────┤
│ Sensors / Jetson / MCU           │
└──────────────────────────────────┘
```

---

# 87. Vision60 전체 Mental Model

최종적으로 Vision60을 다음처럼 볼 수 있다.

```text
                     Vision60

Sensors
│
├── LiDAR ── Ethernet ───────┐
├── Camera ── USB / CSI ─────┤
├── IMU ─────────────────────┤
└── Joint Sensors ───────────┤
                             ▼
                    ┌─────────────────┐
                    │ Jetson / Xavier │
                    │                 │
                    │ ARM CPU         │
                    │ ├── Linux       │
                    │ ├── ROS 2       │
                    │ ├── Drivers     │
                    │ ├── FAST-LIO2   │
                    │ └── Navigation  │
                    │                 │
                    │ NVIDIA GPU      │
                    │ ├── CUDA        │
                    │ ├── Vision      │
                    │ └── TensorRT    │
                    └────────┬────────┘
                             │
                             ▼
                            MCU
                             │
                          CAN / Bus
                             │
                             ▼
                      Motor Controllers
                             │
                             ▼
                           Motors
```

이 그림을 이해하면:

> "센서 데이터가 들어와서 로봇이 움직이기까지 어떤 software/hardware layer를 거치는가?"

를 전체적으로 설명할 수 있다.

---

# 88. Final Checklist

이 학습을 끝낸 뒤 다음 질문에 답할 수 있어야 한다.

### Hardware

```text
CPU와 GPU는 무엇이 다른가?
RAM과 Storage는 무엇이 다른가?
Jetson이 왜 SoC인가?
```

### Architecture

```text
x86_64와 aarch64는 무엇이 다른가?
왜 x86 binary를 Jetson에서 실행할 수 없는가?
```

### Linux

```text
Process와 Service는 무엇이 다른가?
source setup.bash는 무엇을 하는가?
SSH로 접속하면 프로그램은 어디에서 실행되는가?
```

### Jetson

```text
JetPack과 Ubuntu는 무엇이 다른가?
CUDA와 TensorRT는 무엇이 다른가?
tegrastats는 왜 사용하는가?
```

### Interface

```text
Ethernet과 Internet은 무엇이 다른가?
CAN과 ROS 2는 무엇이 다른가?
PCIe와 M.2는 무엇이 다른가?
```

### ROS 2

```text
ROS 2는 protocol인가?
rclcpp → RMW → DDS는 어떻게 연결되는가?
QoS는 왜 필요한가?
```

### GPU

```text
CUDA kernel이란?
Thread / Block / Grid란?
.to("cuda")는 무엇을 하는가?
Latency와 Throughput은 무엇이 다른가?
```

이 질문에 자연스럽게 답할 수 있다면
Jetson 기반 로봇 software architecture의 기본 토대는 잡힌 것이다.

---

# Course Complete

## Jetson & Edge Computing Fundamentals

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
```

이제 다음 단계부터는 기초 개념보다는
실제 Vision60/Jetson 환경을 직접 분석하는 실습 중심 학습으로 넘어갈 수 있다.

추천 다음 과정:

```text
Practical 1
Vision60 Network Architecture 분석

Practical 2
Xavier / Orin System Inspection

Practical 3
ROS 2 Communication Debugging

Practical 4
FAST-LIO2 Runtime Profiling

Practical 5
Jetson CPU/GPU/Memory Profiling

Practical 6
Docker on Jetson

Practical 7
Vision60 전체 Software Architecture 작성
```