---
title: "Chapter 2. ARM vs x86"
importance: 3
---

> **Goal:** Jetson에서 사용하는 ARM CPU와 일반 PC에서 많이 사용하는 x86 CPU의 차이를 이해한다.
> 특히 `aarch64`, `x86_64`, binary compatibility, Docker image, cross compilation이 왜 중요한지 로봇 개발 관점에서 이해한다.

---

# 1. CPU는 아무 명령어나 이해하지 못한다

Chapter 1에서 CPU는 프로그램의 명령을 실행하는 장치라고 배웠다.

하지만 CPU가 우리가 작성한 C++, Rust, Python 코드를 그대로 읽는 것은 아니다.

예를 들어 C++로:

```cpp
int c = a + b;
```

라고 작성했다고 하자.

CPU는 `int`, `a`, `+` 같은 C++ 문법을 직접 이해하지 못한다.

컴파일 과정을 거쳐 CPU가 이해할 수 있는 **Machine Instruction**으로 변환되어야 한다.

```text
C++ Source Code

int c = a + b;

      │
      │ Compiler
      ▼

Machine Instructions

      │
      ▼

CPU
```

그런데 여기서 중요한 문제가 있다.

> 모든 CPU가 같은 Machine Instruction을 이해하는 것은 아니다.

CPU 종류마다 이해하는 명령어 체계가 다를 수 있다.

이것이 ARM과 x86을 이해하는 출발점이다.

---

# 2. ISA란?

ISA는 **Instruction Set Architecture**의 약자다.

쉽게 말하면:

> CPU와 Software 사이에서 약속한 "명령어 규칙"

이다.

CPU가 어떤 명령어를 이해하는지, register는 어떻게 구성되는지, memory를 어떻게 접근하는지 등의 규칙을 정의한다.

전체 구조를 보면:

```text
Application
    │
    ▼
C++ / Rust
    │
    ▼
Compiler
    │
    ▼
Machine Instructions
    │
    │  ← ISA
    ▼
CPU
```

대표적인 ISA 계열이 바로:

```text
x86
ARM
RISC-V
```

등이다.

---

# 3. x86이란?

x86은 Intel에서 시작된 CPU instruction architecture 계열이다.

현재 Desktop PC와 Server에서 매우 널리 사용된다.

대표적인 제조사는:

- Intel
- AMD

이다.

예를 들어 일반적인 Linux workstation이 다음 CPU를 사용한다면:

```text
Intel Core
AMD Ryzen
AMD EPYC
Intel Xeon
```

대부분 x86 계열이다.

현재 64-bit x86 architecture는 보통:

```text
x86_64
```

또는:

```text
AMD64
```

라고 부른다.

---

# 4. ARM이란?

ARM 역시 CPU instruction architecture 계열이다.

특히 다음과 같은 장치에서 많이 사용된다.

- Smartphone
- Tablet
- Embedded System
- Raspberry Pi
- NVIDIA Jetson
- Robot
- Automotive Computer

ARM architecture는 전력 효율이 중요한 시스템에서 널리 사용되어 왔다.

로봇은 배터리로 움직이는 경우가 많기 때문에:

```text
Performance
+
Power Consumption
+
Heat
+
Size
```

를 모두 고려해야 한다.

그래서 ARM 기반 SoC가 Edge Computing에서 많이 사용된다.

---

# 5. ARM과 x86의 핵심 차이

아주 단순화하면:

| x86 | ARM |
|---|---|
| PC / Server에서 매우 흔함 | Mobile / Embedded / Edge에서 매우 흔함 |
| Intel, AMD CPU | Jetson, Raspberry Pi 등의 SoC |
| x86_64 | aarch64 |
| 높은 범용 성능 중심의 역사 | 전력 효율과 SoC 통합에 강점 |

과거에는 흔히:

```text
x86 = CISC
ARM = RISC
```

라고 설명했다.

개념적으로는 도움이 되지만 현대 CPU는 내부 구조가 매우 복잡하기 때문에

> "x86은 복잡하고 ARM은 단순하다"

정도로만 이해하면 지나치게 단순화한 설명이 된다.

실무에서는 오히려 다음이 훨씬 중요하다.

> **ARM용으로 만들어진 프로그램과 x86용으로 만들어진 프로그램은 일반적으로 같은 binary가 아니다.**

---

# 6. x86_64와 aarch64

현재 우리가 자주 만나는 두 architecture 이름은:

```text
x86_64
aarch64
```

이다.

## x86_64

64-bit x86 architecture.

다른 이름으로:

```text
amd64
```

라고도 한다.

예:

```text
Intel PC
AMD PC
많은 Cloud Server
```

---

## aarch64

64-bit ARM architecture를 의미한다.

다른 표현으로:

```text
ARM64
arm64
```

를 볼 수 있다.

Jetson은 대표적인 ARM64 기반 시스템이다.

따라서 Jetson에서 Linux 명령어:

```bash
uname -m
```

을 실행하면 보통:

```text
aarch64
```

가 나온다.

---

# 7. 직접 확인하기

Linux에서 현재 machine architecture를 확인하는 가장 간단한 방법:

```bash
uname -m
```

예를 들어 x86 Linux PC에서는:

```text
x86_64
```

Jetson에서는:

```text
aarch64
```

가 나올 수 있다.

즉:

```text
uname -m
   │
   ▼
현재 Linux가 실행되는 CPU architecture 확인
```

이라고 생각하면 된다.

---

# 8. 왜 Architecture가 중요할까?

다음 상황을 생각해보자.

개발 PC:

```text
Ubuntu PC
AMD CPU
x86_64
```

Robot:

```text
Jetson Orin
ARM CPU
aarch64
```

PC에서 프로그램을 컴파일했다.

```bash
g++ main.cpp -o robot_app
```

그러면 기본적으로 생성되는 binary는 PC architecture를 대상으로 한다.

```text
main.cpp

   │
   │ x86 compiler
   ▼

robot_app
[x86_64 binary]
```

이 파일을 Jetson으로 복사한다.

```text
x86_64 binary
      │
      ▼
Jetson
aarch64
```

그러면 일반적으로 그대로 실행할 수 없다.

왜냐하면 Jetson ARM CPU가 x86 machine instruction을 직접 실행할 수 없기 때문이다.

---

# 9. Source Code와 Binary는 다르다

여기서 매우 중요한 차이가 있다.

C++ source code:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello Robot" << std::endl;
}
```

이 파일 자체는 architecture에 종속되지 않을 수도 있다.

즉 같은 source code를:

```text
                 main.cpp
                    │
           ┌────────┴────────┐
           │                 │
           ▼                 ▼
      x86 compiler      ARM compiler
           │                 │
           ▼                 ▼
      x86_64 binary     aarch64 binary
           │                 │
           ▼                 ▼
          PC              Jetson
```

처럼 각각 컴파일할 수 있다.

따라서:

> **Source code가 동일하다고 binary까지 동일한 것은 아니다.**

---

# 10. "Works on my machine" 문제

로봇 개발에서 자주 발생하는 상황이다.

내 PC에서는:

```bash
./robot_app
```

가 잘 실행된다.

그런데 Jetson으로 옮기면:

```bash
./robot_app
```

실행이 안 된다.

원인은 여러 가지일 수 있지만 architecture mismatch도 대표적인 원인 중 하나다.

확인할 때:

```bash
file robot_app
```

을 사용할 수 있다.

예:

```text
ELF 64-bit ... x86-64
```

또는:

```text
ELF 64-bit ... ARM aarch64
```

같은 정보를 확인할 수 있다.

---

# 11. Docker도 Architecture가 중요하다

Docker를 사용한다고 CPU architecture 문제가 사라지는 것은 아니다.

예를 들어:

```text
Docker Image
linux/amd64
```

이미지를 Jetson에서 실행하려고 한다고 하자.

Jetson은:

```text
linux/arm64
```

환경이다.

구조는:

```text
Docker Image
linux/amd64
     │
     ▼
Jetson
linux/arm64
```

이므로 architecture mismatch 문제가 발생할 수 있다.

---

# 12. amd64와 arm64

Docker에서는 다음 표현을 매우 자주 본다.

```text
linux/amd64
linux/arm64
```

대략 다음처럼 연결하면 된다.

```text
x86_64  ≈ amd64
aarch64 ≈ arm64
```

예:

```text
Desktop Ubuntu
Architecture: x86_64
Docker platform: linux/amd64
```

```text
Jetson
Architecture: aarch64
Docker platform: linux/arm64
```

이 관계는 꼭 기억해두는 것이 좋다.

---

# 13. Multi-Architecture Docker Image

일부 Docker image는 여러 architecture를 지원한다.

예를 들어 하나의 image 이름 아래:

```text
my_robot_image

├── linux/amd64
└── linux/arm64
```

버전이 모두 존재할 수 있다.

이런 image를 **multi-platform / multi-architecture image**라고 한다.

사용자는 같은 image 이름을 사용하더라도 Docker가 현재 platform에 맞는 image를 선택할 수 있다.

```text
                Docker Image
                     │
             ┌───────┴───────┐
             │               │
        linux/amd64      linux/arm64
             │               │
             ▼               ▼
          x86 PC           Jetson
```

로봇용 Docker image를 만들 때 중요한 개념이다.

---

# 14. Jetson에서는 한 가지가 더 중요하다

Jetson에서는 단순히:

```text
ARM CPU
```

만 고려하면 끝나지 않는다.

Jetson에는 NVIDIA GPU가 있기 때문이다.

따라서 실제 환경에서는:

```text
CPU Architecture
+
CUDA Version
+
JetPack Version
+
Driver
+
Library Version
```

까지 호환성을 확인해야 한다.

예를 들어:

```text
Jetson Orin
   │
   ├── ARM64 CPU
   ├── NVIDIA GPU
   ├── JetPack
   ├── CUDA
   ├── cuDNN
   └── TensorRT
```

가 하나의 software stack으로 연결되어 있다.

그래서 일반 Ubuntu PC용 Docker image를 Jetson에 그대로 가져오는 것이 항상 가능한 것은 아니다.

---

# 15. Cross Compilation

그렇다면 x86 PC에서 Jetson용 프로그램을 만들 수는 없을까?

가능하다.

이를 위해 사용하는 방법 중 하나가 **Cross Compilation**이다.

Cross Compilation은:

> 프로그램을 컴파일하는 컴퓨터와 실제 프로그램이 실행될 CPU architecture가 다른 컴파일 방식

이다.

예:

```text
x86_64 Development PC
        │
        │ Cross Compiler
        ▼
aarch64 Binary
        │
        ▼
   Jetson Orin
```

즉:

```text
Compile machine = x86_64
Target machine  = aarch64
```

가 가능하다.

---

# 16. Native Compilation vs Cross Compilation

## Native Compilation

Jetson에서 직접 compile:

```text
Jetson
aarch64

source
  │
  ▼
compile
  │
  ▼
aarch64 binary
```

이것을 native compilation이라고 한다.

예:

```bash
g++ main.cpp -o robot_app
```

를 Jetson에서 직접 실행한다.

장점:

- 단순하다.
- 환경 차이가 적다.
- 디버깅하기 편하다.

단점:

- Jetson의 CPU 성능이나 storage에 따라 대규모 build가 느릴 수 있다.

---

## Cross Compilation

강력한 PC에서 Jetson용 binary 생성:

```text
PC
x86_64
   │
   │ Cross Compile
   ▼
aarch64 binary
   │
   ▼
Jetson
```

장점:

- 강력한 workstation에서 빠르게 build 가능
- CI/CD와 결합하기 좋음

단점:

- Toolchain 설정이 복잡해질 수 있음
- Target library와 dependency를 맞춰야 함

---

# 17. Compiler와 Target Architecture

Compiler도 target architecture를 알아야 한다.

일반적으로 PC에서:

```bash
g++
```

를 실행하면 현재 PC를 위한 binary를 생성한다.

ARM64 target compiler의 이름에서는 다음과 같은 표현을 볼 수 있다.

```text
aarch64-linux-gnu
```

예:

```bash
aarch64-linux-gnu-g++
```

이름을 분해하면:

```text
aarch64
   │
   └── Target Architecture

linux
   │
   └── Target OS

gnu
   │
   └── GNU Toolchain
```

정도로 이해할 수 있다.

---

# 18. Python은 괜찮은가?

Python source code는 조금 다르다.

예를 들어:

```python
print("Hello Robot")
```

같은 Python code 자체는 일반적으로 CPU machine instruction binary가 아니다.

Python interpreter가 실행한다.

```text
Python Source
     │
     ▼
Python Interpreter
     │
     ▼
CPU
```

따라서 동일한 `.py` 파일을 x86과 ARM에서 실행할 수 있는 경우가 많다.

하지만 여기에도 함정이 있다.

---

# 19. Python Library는 Architecture에 영향을 받을 수 있다

예를 들어:

```python
import numpy
import torch
```

를 사용한다고 하자.

Python 코드 자체는 같더라도 NumPy, PyTorch 같은 package 내부에는 C/C++/CUDA로 compile된 native binary가 포함될 수 있다.

따라서:

```text
Python Source
      │
      ├── Pure Python
      │
      └── Native Library
             │
             ├── x86_64
             └── aarch64
```

문제가 발생할 수 있다.

그래서:

> "Python이니까 architecture는 신경 쓰지 않아도 된다"

는 것은 틀린 생각이다.

특히 Jetson에서 PyTorch, CUDA, TensorRT 등을 설치할 때 architecture와 JetPack compatibility가 중요하다.

---

# 20. Mac도 확인해보자

최근 Apple Silicon Mac은:

```text
M1
M2
M3
M4
...
```

와 같은 ARM 기반 processor를 사용한다.

터미널에서:

```bash
uname -m
```

을 실행하면 보통:

```text
arm64
```

가 나온다.

즉:

```text
Apple Silicon Mac → ARM64
Jetson            → ARM64
```

이다.

하지만 둘 다 ARM64라고 해서 binary를 그대로 공유할 수 있는 것은 아니다.

왜냐하면 OS와 ABI, library 환경 등이 다르기 때문이다.

```text
Mac

ARM64
+
macOS

vs

Jetson

ARM64
+
Linux
```

따라서:

```text
Same CPU Architecture
≠
Same Executable Environment
```

이다.

---

# 21. OS와 Architecture는 다른 개념이다

이 부분도 중요하다.

Ubuntu와 ARM은 같은 종류의 이름이 아니다.

```text
Ubuntu
→ Operating System

ARM
→ CPU Architecture
```

따라서 다음과 같은 조합이 가능하다.

```text
Ubuntu + x86_64
Ubuntu + ARM64

macOS + ARM64
macOS + x86_64
```

즉 컴퓨터 환경을 볼 때 최소한:

```text
Hardware Architecture
        +
Operating System
```

을 따로 생각해야 한다.

---

# 22. Vision60 / Jetson 관점에서 보기

우리가 실제로 개발하는 상황을 생각해보자.

```text
Development Computer
        │
        │ Git
        ▼
Source Code
        │
        ▼
Vision60 Jetson
        │
        ├── ARM64 CPU
        ├── NVIDIA GPU
        ├── Ubuntu
        ├── ROS 2
        ├── CUDA
        └── FAST-LIO2
```

Git repository의 source code를 clone하는 것은 architecture와 크게 관계없을 수 있다.

하지만:

```text
Build
Docker
Library Installation
Binary Deployment
```

단계부터 architecture가 매우 중요해진다.

---

# 23. ROS 2에서도 Architecture가 중요하다

ROS 2 package가 source code 형태라면 Jetson에서 직접 build할 수 있다.

예:

```bash
colcon build
```

그러면 Jetson에서:

```text
C++ Source
    │
    │ colcon / CMake
    ▼
ARM64 Binary
```

가 만들어진다.

그래서 source repository를 Jetson에 clone한 뒤 직접 `colcon build`하는 방식을 많이 볼 수 있다.

반면 x86 PC에서 이미 build한 ROS 2 executable을 Jetson으로 단순 복사하면 architecture 문제를 만날 수 있다.

---

# 24. 실무에서 확인할 명령어

## 현재 CPU Architecture

```bash
uname -m
```

예:

```text
x86_64
```

또는:

```text
aarch64
```

---

## 더 자세한 CPU 정보

```bash
lscpu
```

확인할 수 있는 정보:

```text
Architecture
CPU count
Core
Thread
Model
```

---

## Binary Architecture 확인

```bash
file <binary>
```

예:

```bash
file robot_app
```

---

## Ubuntu Package Architecture 확인

```bash
dpkg --print-architecture
```

예:

```text
amd64
```

또는:

```text
arm64
```

---

## Docker Architecture 확인

```bash
docker info
```

또는 image 정보를 확인할 때:

```bash
docker image inspect <image>
```

등을 사용할 수 있다.

---

# 25. 용어 정리

| 용어 | 의미 |
|---|---|
| ISA | CPU가 이해하는 명령어 체계 |
| x86 | Intel에서 시작된 CPU architecture 계열 |
| x86_64 | 64-bit x86 |
| amd64 | 일반적으로 x86_64를 가리키는 이름 |
| ARM | Embedded/Mobile/Edge에서 널리 사용되는 architecture 계열 |
| ARM64 | 64-bit ARM |
| aarch64 | ARM의 64-bit execution architecture를 나타내는 표현 |
| Binary | CPU가 실행할 수 있도록 만들어진 executable code |
| Native Compilation | 실행할 machine과 같은 architecture에서 compile |
| Cross Compilation | 다른 architecture를 위한 binary를 compile |
| Multi-Arch Image | 여러 CPU architecture를 지원하는 container image |

---

# 26. 가장 중요한 연결 관계

전체 내용을 한 그림으로 연결하면:

```text
Source Code
    │
    ▼
Compiler
    │
    │ Target ISA
    ▼
Binary
    │
    ├──────────────┐
    ▼              ▼
 x86_64         aarch64
    │              │
    ▼              ▼
Intel/AMD PC     Jetson
```

Docker까지 추가하면:

```text
Development Environment
        │
        ▼
     Software
        │
        ├── linux/amd64 → x86_64 machine
        │
        └── linux/arm64 → ARM64 machine
                            │
                            ▼
                        Jetson Orin
```

---

# 27. 오늘의 핵심

다음 네 가지는 꼭 기억한다.

### 1.

```text
x86_64 ≈ amd64
```

### 2.

```text
aarch64 ≈ arm64
```

### 3.

```text
Same Source Code
≠
Same Binary
```

### 4.

Jetson에서 software를 설치할 때는 항상:

```text
Architecture
+
OS
+
JetPack
+
CUDA
+
Library Version
```

을 확인해야 한다.

---

# 28. Mini Practice

현재 사용 중인 Linux PC나 Jetson에서 다음을 실행해본다.

```bash
uname -m
```

```bash
lscpu
```

```bash
dpkg --print-architecture
```

그리고 각각의 결과가:

```text
x86_64 / amd64
```

인지:

```text
aarch64 / arm64
```

인지 확인한다.

실행 파일이 있다면:

```bash
file <binary>
```

도 실행해본다.

---

# Next Chapter

## Chapter 3. Linux for Edge Computers

다음 Chapter에서는 Jetson에 SSH로 접속했을 때 실제로 가장 많이 사용하는 Linux 개념을 다룬다.

- Linux filesystem은 어떻게 생겼는가?
- `/`, `/home`, `/etc`, `/opt`, `/dev`, `/proc`는 무엇인가?
- Process와 Service는 무엇이 다른가?
- `ps`, `top`, `htop`은 무엇을 보여주는가?
- `sudo`는 정확히 무엇인가?
- `apt`는 무엇을 하는가?
- Environment Variable은 무엇인가?
- `PATH`는 왜 필요한가?
- `source setup.bash`는 대체 무엇을 하는가?
- Jetson에 SSH로 접속한다는 것은 무슨 뜻인가?

Chapter 3부터는 실제 Jetson/Ubuntu terminal에서 보는 것들이 본격적으로 연결된다.