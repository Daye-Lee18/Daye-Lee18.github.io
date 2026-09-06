---
title: "Chapter 9. Docker on Jetson"
importance: 10
---


> **Goal:** Docker container가 무엇인지 이해하고,
> Jetson에서 ROS 2, CUDA, TensorRT 환경을 container로 관리하는 방법을 이해한다.
> 또한 ARM64, GPU runtime, network, device access가 container에서 왜 중요한지 이해한다.

---

# 1. Docker는 왜 사용할까?

로봇 software는 dependency가 많다.

예:

```text
Ubuntu
ROS 2
Python
C++
CUDA
cuDNN
TensorRT
PyTorch
Sensor SDK
FAST-LIO2
```

이 중 하나라도 version이 다르면 프로그램이 안 돌아갈 수 있다.

예:

```text
PC A
ROS 2 Humble
CUDA 12.x
Python 3.10

PC B
ROS 2 Humble
CUDA 11.x
Python 3.8
```

같은 source code인데 한 곳에서는 되고 다른 곳에서는 안 될 수 있다.

Docker는 이런 software environment를 묶어서 재현하기 쉽게 만든다.

---

# 2. Container란?

Container는 host OS 위에서
격리된 user-space environment를 실행하는 기술이다.

구조:

```text
Application
Libraries
Dependencies
      │
      ▼
Container
      │
      ▼
Host Linux Kernel
      │
      ▼
Hardware
```

중요:

> Container는 별도의 Linux kernel을 가지는 완전한 VM과 다르다.

---

# 3. Virtual Machine과 Container 차이

VM:

```text
Application
Libraries
Guest OS
Guest Kernel
Hypervisor
Host OS
Hardware
```

Container:

```text
Application
Libraries
Container Runtime
Host Linux Kernel
Hardware
```

즉 container는 host kernel을 공유한다.

그래서 일반적으로 VM보다 가볍고 빠르게 시작한다.

---

# 4. Docker Image와 Container

이 둘은 반드시 구분해야 한다.

## Image

실행 환경의 template.

예:

```text
Ubuntu
ROS 2
CUDA
Python
App
```

를 묶은 read-only template.

## Container

Image를 실제로 실행한 instance.

```text
Image
  │
  │ docker run
  ▼
Container
```

---

# 5. Image와 Container 관계

비유:

```text
Image = 설계도
Container = 실제 실행된 집
```

하나의 image로 여러 container를 만들 수 있다.

```text
Image
├── Container A
├── Container B
└── Container C
```

---

# 6. Dockerfile

Docker image를 어떻게 만들지 정의하는 파일이:

```text
Dockerfile
```

이다.

예:

```dockerfile
FROM ubuntu:22.04

RUN apt update && apt install -y python3

COPY app.py /app/app.py

CMD ["python3", "/app/app.py"]
```

---

# 7. `FROM`

Base image를 정한다.

예:

```dockerfile
FROM ubuntu:22.04
```

의미:

> Ubuntu 22.04 환경을 기반으로 image를 만든다.

---

# 8. `RUN`

Image build 중 command를 실행한다.

예:

```dockerfile
RUN apt update && apt install -y git
```

이 결과가 image layer에 저장된다.

---

# 9. `COPY`

Host의 파일을 image 안으로 복사한다.

예:

```dockerfile
COPY src/ /workspace/src/
```

---

# 10. `WORKDIR`

Container 내부의 working directory를 정한다.

예:

```dockerfile
WORKDIR /workspace
```

---

# 11. `CMD`

Container가 시작될 때 기본적으로 실행할 command를 정한다.

예:

```dockerfile
CMD ["bash"]
```

또는:

```dockerfile
CMD ["python3", "app.py"]
```

---

# 12. Docker Image Build

Dockerfile로 image를 만든다.

```bash
docker build -t robot-app .
```

여기서:

```text
-t
→ tag

robot-app
→ image name

.
→ Docker build context
```

이다.

---

# 13. Docker Container 실행

```bash
docker run robot-app
```

구조:

```text
Image
robot-app
   │
   ▼
docker run
   │
   ▼
Container
```

---

# 14. Interactive Container

터미널로 직접 들어가려면:

```bash
docker run -it ubuntu:22.04 bash
```

여기서:

```text
-i
→ interactive

-t
→ terminal
```

이다.

---

# 15. Container 안에서는 별도 Filesystem처럼 보인다

예:

```bash
ls /
```

를 container 안에서 실행하면 container filesystem이 보인다.

```text
/
├── bin
├── etc
├── home
├── usr
└── ...
```

하지만 host filesystem과 기본적으로 분리되어 있다.

---

# 16. Container를 종료하면 데이터는?

Container 내부에서 만든 data는 container에 존재한다.

Container를 삭제하면 같이 사라질 수 있다.

그래서 source code, logs, maps 등을 보존하려면:

```text
Volume
Bind Mount
```

를 사용한다.

---

# 17. Bind Mount

Host directory를 container 안에 연결한다.

예:

```bash
docker run -it \
  -v ~/vision60_ws:/workspace \
  ubuntu:22.04
```

구조:

```text
Host
~/vision60_ws
      │
      │ bind mount
      ▼
Container
/workspace
```

---

# 18. Bind Mount 장점

Host에서 source code를 수정하면 container 안에서도 바로 보인다.

```text
Host VSCode
   │
   ▼
Source Code
   │
   ▼
Bind Mount
   │
   ▼
Container Build
```

개발 환경에서 매우 편리하다.

---

# 19. Docker Volume

Docker가 관리하는 persistent storage다.

```text
Docker Volume
     │
     ├── Container A
     └── Container B
```

Bind mount는 host path를 직접 지정하지만,
volume은 Docker가 저장 위치를 관리한다.

---

# 20. Container Process

Container도 결국 host Linux에서 실행되는 process다.

예:

```text
Host Linux

PID 1000 docker
PID 1500 ros2 node
PID 1501 python
```

container가 완전히 별도 컴퓨터처럼 보이지만
실제로는 host kernel 위의 process다.

---

# 21. Namespace

Linux namespace는 container isolation의 핵심 기술 중 하나다.

분리할 수 있는 것:

```text
Process
Network
Mount
Hostname
User
```

그래서 container 안에서는 자기만의 process tree와 network처럼 보일 수 있다.

---

# 22. Cgroups

Cgroups는 resource를 제한하고 관리한다.

예:

```text
CPU
RAM
```

제한 가능.

즉 Docker는 대략:

```text
Namespaces
+
Cgroups
+
Filesystem Layers
```

를 활용한다고 보면 된다.

---

# 23. Docker Network

기본적으로 container는 별도의 virtual network를 사용할 수 있다.

예:

```text
Host

eth0
 │
Docker Bridge
 │
docker0
 │
Container
```

container에는 별도 IP가 할당될 수 있다.

---

# 24. ROS 2에서 Docker Network가 문제될 수 있는 이유

ROS 2 DDS discovery는 multicast나 network interface 설정에 영향을 받을 수 있다.

구조:

```text
Host ROS 2
      │
      X
Docker Bridge
      │
Container ROS 2
```

설정에 따라 서로 discovery되지 않을 수 있다.

---

# 25. `--network host`

Linux에서는 container가 host network stack을 직접 사용하게 할 수 있다.

```bash
docker run --network host ...
```

구조:

```text
Container
   │
   ▼
Host Network
eth0 / wlan0
```

ROS 2 개발에서는 이 방법을 많이 볼 수 있다.

---

# 26. `--network host`의 장점

ROS 2 node가 host와 같은 network interface를 사용한다.

```text
Host ROS 2
      │
      │ same network namespace
      ▼
Container ROS 2
```

DDS discovery 설정이 단순해질 수 있다.

---

# 27. 단점

Network isolation이 줄어든다.

즉 container가 host network에 직접 접근한다.

따라서:

```text
Security
Port conflict
Isolation
```

을 고려해야 한다.

---

# 28. Container에서 USB Device 접근

기본적으로 container가 host의 모든 hardware에 자동 접근할 수 있는 것은 아니다.

예:

```text
Host
/dev/ttyUSB0
      │
      X
Container
```

device를 전달하려면:

```bash
docker run \
  --device=/dev/ttyUSB0 \
  ...
```

같은 방식이 가능하다.

---

# 29. `/dev`와 Container

Chapter 3에서 배운 것처럼:

```text
/dev/ttyUSB0
/dev/video0
```

는 Linux device file이다.

Container에 해당 device를 노출하면
container 안의 application이 hardware를 사용할 수 있다.

---

# 30. Camera 전달 예

예:

```bash
docker run \
  --device=/dev/video0 \
  ...
```

구조:

```text
USB Camera
   │
   ▼
Host Linux
/dev/video0
   │
   ▼
Container
   │
   ▼
ROS Camera Driver
```

---

# 31. `--privileged`

가끔:

```bash
docker run --privileged ...
```

를 볼 수 있다.

이 옵션은 container에 매우 강한 hardware/system access 권한을 준다.

개발할 때 편하지만:

> 필요 이상으로 권한이 너무 크다.

따라서 production에서는 필요한 device/capability만 구체적으로 주는 것이 좋다.

---

# 32. CAN과 Docker

Host에서:

```text
can0
```

가 있다면 container에서 CAN을 사용하려면
network namespace 설정을 고려해야 한다.

`--network host`를 사용하면 host의 CAN interface도 같은 network namespace에서 볼 수 있는 경우가 있다.

환경에 따라 capability와 device 설정이 추가로 필요할 수 있다.

---

# 33. Jetson GPU와 Docker

Jetson container에서 GPU를 사용하려면 단순한 Ubuntu container만으로는 충분하지 않을 수 있다.

구조:

```text
Container
PyTorch / TensorRT
      │
      ▼
NVIDIA Runtime
      │
      ▼
Host Jetson Driver
      │
      ▼
Jetson GPU
```

---

# 34. NVIDIA Container Runtime

NVIDIA GPU를 container 안에서 사용하도록
필요한 driver/library를 연결해주는 runtime이다.

핵심 개념:

> GPU driver 자체는 host가 관리하고,
> container가 이를 사용할 수 있게 연결한다.

---

# 35. Driver는 Container 안에 완전히 독립적으로 넣지 않는다

일반적으로:

```text
Host
NVIDIA Driver
      │
      ▼
Container
CUDA User-space Libraries
```

형태다.

Kernel driver는 host kernel과 연결되어 있다.

Chapter 3의:

```text
Container shares Host Kernel
```

과 연결된다.

---

# 36. Jetson과 Desktop CUDA Docker는 다를 수 있다

Desktop NVIDIA GPU:

```text
x86_64
+
Discrete GPU
```

Jetson:

```text
ARM64
+
Integrated NVIDIA GPU
+
JetPack / L4T
```

이므로 image compatibility를 반드시 확인해야 한다.

---

# 37. `linux/amd64` vs `linux/arm64`

Chapter 2에서 배운 것처럼 Docker image에도 architecture가 있다.

```text
linux/amd64
→ x86_64

linux/arm64
→ ARM64 / aarch64
```

Jetson에서는 보통:

```text
linux/arm64
```

image가 필요하다.

---

# 38. x86 Docker Image를 Jetson에서 실행하면?

예:

```text
Image
linux/amd64

Host
Jetson ARM64
```

architecture mismatch가 발생한다.

일부 emulation을 사용할 수는 있지만
성능과 compatibility 문제가 있을 수 있다.

---

# 39. Multi-Arch Image

하나의 image name 아래 여러 architecture를 제공할 수 있다.

```text
robot-image

├── linux/amd64
└── linux/arm64
```

그러면 개발 PC와 Jetson에서 같은 image tag를 사용할 수 있다.

---

# 40. Multi-Stage Build

Docker image size를 줄이기 위해
build 환경과 runtime 환경을 분리할 수 있다.

예:

```dockerfile
FROM ubuntu:22.04 AS builder

RUN ...

FROM ubuntu:22.04

COPY --from=builder /app/bin /app/bin
```

구조:

```text
Builder Image
compiler
headers
tools
      │
      ▼
Runtime Image
binary only
```

---

# 41. 왜 Image Size가 중요할까?

Jetson은 storage와 network가 제한될 수 있다.

큰 image는:

```text
Download 느림
Deploy 느림
Storage 많이 사용
Update 느림
```

문제가 있다.

---

# 42. Layer Cache

Docker image는 여러 layer로 구성된다.

예:

```text
FROM Ubuntu
RUN apt install
COPY source
RUN build
```

변경되지 않은 layer는 cache를 사용할 수 있다.

그래서 Dockerfile 순서가 build 속도에 영향을 줄 수 있다.

---

# 43. 좋은 Dockerfile 순서

자주 안 바뀌는 dependency는 위쪽:

```dockerfile
FROM ...
RUN apt install ...
RUN pip install ...
```

자주 바뀌는 source는 아래쪽:

```dockerfile
COPY src/ ...
```

에 두면 build cache를 더 잘 활용할 수 있다.

---

# 44. ROS 2 Docker Image

ROS 2 base image를 사용할 수도 있다.

개념:

```dockerfile
FROM ros:humble

RUN apt update && apt install -y ...
```

그러면 ROS 2 환경을 처음부터 직접 설치할 필요가 줄어든다.

단 Jetson에서는 architecture와 GPU/JetPack compatibility를 확인해야 한다.

---

# 45. `source` 문제

Container에서도 ROS 2를 사용하려면:

```bash
source /opt/ros/humble/setup.bash
```

가 필요할 수 있다.

Dockerfile이나 entrypoint에서 자동으로 source하도록 만들 수 있다.

---

# 46. Entrypoint

Container가 시작할 때 특정 script를 항상 실행하도록 할 수 있다.

예:

```bash
#!/bin/bash

source /opt/ros/humble/setup.bash

exec "$@"
```

이 script를 entrypoint로 사용하면
container 안에서 ROS environment가 자동 설정된다.

---

# 47. `exec "$@"`

Chapter 3에서 배운 `exec`가 다시 나온다.

```bash
exec "$@"
```

는 entrypoint shell을
실제로 실행할 application으로 교체한다.

이렇게 하면 signal 전달이 더 자연스럽다.

예:

```text
Docker
   │
Entrypoint Bash
   │
exec
   ▼
ROS 2 Process
```

---

# 48. Signal 전달이 중요한 이유

Container 종료 시:

```bash
docker stop
```

은 container의 main process에 signal을 보낸다.

ROS 2 process가 main process이면
정상 종료 처리를 하기 쉽다.

---

# 49. PID 1

Container 안에서 main process는 보통 PID 1이다.

PID 1은 signal handling과 child process 정리에서 특별한 역할이 있다.

그래서 entrypoint에서:

```bash
exec ros2 launch ...
```

를 사용하는 패턴을 자주 볼 수 있다.

---

# 50. Environment Variable 전달

Docker 실행 시:

```bash
docker run \
  -e ROS_DOMAIN_ID=123 \
  ...
```

처럼 environment variable을 전달할 수 있다.

Container 안에서:

```bash
echo $ROS_DOMAIN_ID
```

로 확인 가능하다.

---

# 51. `.env`

환경 변수를 파일로 관리할 수도 있다.

예:

```text
ROS_DOMAIN_ID=123
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Docker Compose 등에서 사용할 수 있다.

---

# 52. ROS 2와 CycloneDDS Configuration

Container 안에서도:

```bash
CYCLONEDDS_URI
```

같은 설정이 필요할 수 있다.

예:

```text
Host
cyclonedds.xml
    │
    ▼
Bind Mount
    │
    ▼
Container
```

---

# 53. Docker에서 Hostname

Container는 기본적으로 별도 hostname을 가진다.

확인:

```bash
hostname
```

필요하면:

```bash
docker run --hostname robot-container ...
```

처럼 지정할 수 있다.

---

# 54. Port Mapping

Web application 같은 경우:

```bash
docker run -p 8080:80 ...
```

를 사용한다.

의미:

```text
Host 8080
   │
   ▼
Container 80
```

---

# 55. ROS 2에서는 Port Mapping만으로 충분하지 않을 수 있다

ROS 2/DDS는 여러 dynamic port와 discovery mechanism을 사용할 수 있다.

그래서 단순히:

```bash
-p 1234:1234
```

하나만 열어주는 방식보다
`--network host`를 사용하는 예가 많다.

---

# 56. Docker Compose

여러 container를 함께 관리할 때:

```text
docker compose
```

를 사용할 수 있다.

예:

```text
LiDAR Container
SLAM Container
AI Container
```

를 하나의 configuration으로 관리할 수 있다.

---

# 57. Compose 예시

```yaml
services:

  slam:
    image: vision60-slam
    network_mode: host
    environment:
      ROS_DOMAIN_ID: 123

  perception:
    image: vision60-perception
    network_mode: host
```

---

# 58. Container를 기능별로 나누기

예:

```text
Container A
LiDAR Driver

Container B
FAST-LIO2

Container C
AI Perception
```

장점:

```text
Dependency isolation
Independent update
Fault isolation
```

---

# 59. 너무 잘게 나누면 생기는 문제

Container가 많아질수록:

```text
Network complexity
Deployment complexity
Debugging complexity
Image management
Startup ordering
```

가 증가한다.

따라서 적절한 분리가 중요하다.

---

# 60. 하나의 Container에 모두 넣는 경우

예:

```text
vision60-autonomy container

├── ROS 2
├── LiDAR Driver
├── FAST-LIO2
├── Nav2
└── Config
```

장점:

```text
단순한 deployment
```

단점:

```text
Dependency coupling
Image size 증가
부분 update 어려움
```

---

# 61. Source Code를 Image에 넣을까 Mount할까?

개발 단계:

```text
Bind Mount
```

가 편하다.

```text
Host source
      │
      ▼
Container build/run
```

Production:

```text
Source / Binary를 image에 포함
```

하는 방식이 reproducibility에 유리할 수 있다.

---

# 62. Development vs Production

Development:

```text
-v source:/workspace
Interactive shell
Debug tools
```

Production:

```text
Immutable image
Minimal dependencies
Fixed version
Auto start
```

로 구분하는 것이 좋다.

---

# 63. Tag

Image version을 tag로 관리한다.

예:

```text
vision60-slam:latest
vision60-slam:v1.0.0
vision60-slam:2026-09-05
```

Production에서는 `latest`만 사용하는 것보다
명시적인 version tag가 reproducibility에 좋다.

---

# 64. Registry

Docker image를 저장하고 배포하는 server를 registry라고 한다.

예:

```text
Docker Hub
GitHub Container Registry
Private Registry
AWS ECR
```

구조:

```text
Developer
   │
   ▼
Build Image
   │
   ▼
Registry
   │
   ▼
Robot
docker pull
```

---

# 65. Robot Deployment

예:

```text
CI Server
   │
   ▼
Build ARM64 Image
   │
   ▼
Registry
   │
   ▼
Jetson
   │
docker pull
   │
docker run
```

이런 식으로 robot software update pipeline을 만들 수 있다.

---

# 66. Version Pinning

Docker에서도 version을 정확히 고정하는 것이 중요하다.

좋지 않은 예:

```dockerfile
RUN pip install torch
```

시간이 지나면 다른 version이 설치될 수 있다.

더 재현 가능한 방식:

```text
Fixed base image
Fixed package versions
Fixed git commit
```

을 사용한다.

---

# 67. JetPack Version과 Container

Jetson에서는 host JetPack/L4T와 container의 CUDA stack compatibility가 중요하다.

개념:

```text
Host
JetPack / Driver
      │
      ▼
Container
CUDA / TensorRT user-space
```

version mismatch가 크면 GPU runtime 문제가 발생할 수 있다.

---

# 68. Jetson Container 확인할 것

Jetson용 image를 사용할 때:

```text
1. ARM64인가?
2. JetPack/L4T 호환되는가?
3. CUDA version 맞는가?
4. TensorRT/cuDNN 필요한가?
5. ROS 2 version 맞는가?
```

확인한다.

---

# 69. Image 안에서 Architecture 확인

Container 안:

```bash
uname -m
```

실행.

보통 Jetson에서는:

```text
aarch64
```

가 나온다.

Container가 host kernel을 공유하기 때문이다.

---

# 70. Container 안에서 GPU 확인

PyTorch가 있다면:

```python
import torch
print(torch.cuda.is_available())
```

확인.

`False`라면:

```text
Runtime
CUDA library
Container image
JetPack compatibility
```

등을 확인해야 한다.

---

# 71. Docker에서 Device Permission 문제

예:

```text
/dev/ttyUSB0 exists

but

Permission denied
```

Container에서도 user/group permission이 영향을 줄 수 있다.

무작정 root container로 해결하기보다
device group과 permission을 이해하는 것이 좋다.

---

# 72. User ID 문제

Bind mount를 사용하면 host와 container의 UID/GID가 다를 때
파일 owner 문제가 생길 수 있다.

예:

```text
Host user
UID 1000

Container root
UID 0
```

Container에서 생성한 파일이 host에서 root-owned가 될 수 있다.

---

# 73. `--user`

필요하면:

```bash
docker run --user 1000:1000 ...
```

처럼 container process를 특정 UID/GID로 실행할 수 있다.

---

# 74. Rootless와 Security

Container를 항상 root로 실행하는 것은 편하지만
보안 관점에서는 권한을 최소화하는 것이 좋다.

원칙:

```text
Least Privilege
```

필요한 권한만 준다.

---

# 75. Secret를 Image에 넣지 않는다

다음은 Dockerfile에 넣으면 안 된다.

```text
Password
SSH Private Key
API Token
Cloud Credential
```

Image layer에 남을 수 있다.

Secret은 runtime에 안전하게 전달해야 한다.

---

# 76. Robot에서 Docker가 좋은 이유

로봇 여러 대에 같은 environment를 배포하기 쉽다.

```text
Robot A
Robot B
Robot C
```

모두 같은 image:

```text
vision60-autonomy:v1.2.0
```

를 사용하면 environment 차이를 줄일 수 있다.

---

# 77. "Works on my machine" 문제 감소

Docker를 사용하면:

```text
Developer PC
      │
Same Image
      ▼
CI
      │
Same Image
      ▼
Robot
```

방식으로 environment를 통일할 수 있다.

단 hardware architecture가 다르면 image도 multi-arch로 준비해야 한다.

---

# 78. Docker가 모든 문제를 해결하지는 않는다

Docker로도 다음 문제는 남는다.

```text
Kernel
Driver
Hardware
JetPack
Network
Device
Architecture
```

즉:

```text
Docker container works everywhere
```

는 아니다.

특히 Jetson에서는 host driver/JetPack dependency가 매우 중요하다.

---

# 79. Vision60 Example

예를 들어 Orin에서:

```text
Host Ubuntu / JetPack
        │
        ▼
Docker
        │
        ├── ROS 2 Humble
        ├── FAST-LIO2
        ├── CycloneDDS
        └── Vision AI
```

구성할 수 있다.

---

# 80. LiDAR Docker Pipeline

```text
LiDAR
   │
 Ethernet
   │
   ▼
Host Network
   │
   ▼
Docker --network host
   │
   ▼
LiDAR Driver Node
   │
   ▼
ROS 2 PointCloud2
```

---

# 81. Camera + GPU Docker Pipeline

```text
Camera
   │
/dev/video0
   │
   ▼
Host Linux
   │
 --device
   │
   ▼
Container
   │
   ▼
PyTorch / TensorRT
   │
   ▼
Jetson GPU
```

---

# 82. ROS 2 Multi-Container

```text
Container A
LiDAR Node
      │
      │ ROS 2
      ▼
Container B
FAST-LIO2
      │
      ▼
Container C
Navigation
```

같은 ROS_DOMAIN_ID와 network 설정이 필요하다.

---

# 83. Debugging 순서

Container 안에서 ROS 2 sensor가 안 보인다면:

```text
1. Host hardware 보임?
       ↓
2. Host driver/interface 정상?
       ↓
3. Device가 container에 전달됨?
       ↓
4. Container network 정상?
       ↓
5. ROS_DOMAIN_ID?
       ↓
6. RMW/DDS?
       ↓
7. Topic 있음?
       ↓
8. QoS?
```

순서로 본다.

---

# 84. GPU Container Debugging

GPU가 안 보인다면:

```text
1. Host GPU 정상?
       ↓
2. JetPack 정상?
       ↓
3. NVIDIA runtime 정상?
       ↓
4. Image ARM64?
       ↓
5. CUDA compatibility?
       ↓
6. PyTorch/TensorRT build?
```

순서로 본다.

---

# 85. 자주 쓰는 Docker 명령어

Image 목록:

```bash
docker images
```

Container 목록:

```bash
docker ps
```

종료된 것 포함:

```bash
docker ps -a
```

Container 실행:

```bash
docker run ...
```

Container 내부 shell:

```bash
docker exec -it <container> bash
```

Container 로그:

```bash
docker logs <container>
```

정지:

```bash
docker stop <container>
```

삭제:

```bash
docker rm <container>
```

Image 삭제:

```bash
docker rmi <image>
```

---

# 86. `docker exec`

이미 실행 중인 container 안에 command를 실행한다.

예:

```bash
docker exec -it vision60 bash
```

구조:

```text
Running Container
      │
      ▼
New Bash Process
```

container를 새로 만드는 것이 아니다.

---

# 87. `docker logs`

Container의 stdout/stderr를 확인한다.

```bash
docker logs vision60
```

실시간:

```bash
docker logs -f vision60
```

로봇 service debugging에서 매우 유용하다.

---

# 88. Restart Policy

Robot reboot 후 container를 자동으로 다시 실행하고 싶을 수 있다.

예:

```bash
--restart unless-stopped
```

같은 restart policy를 사용할 수 있다.

---

# 89. Docker와 systemd

systemd service에서 Docker container를 실행하는 방식도 가능하다.

구조:

```text
Boot
 ↓
systemd
 ↓
Docker
 ↓
Robot Container
 ↓
ROS 2
```

Production robot에서 자주 고려하는 구조다.

---

# 90. Health Check

Container가 살아 있다고 application까지 정상이라는 뜻은 아니다.

예:

```text
Container running
but ROS node crashed
```

가능하다.

그래서 health check를 설계할 수 있다.

예:

```text
ROS node alive?
Sensor data arriving?
Heartbeat present?
```

---

# 91. Container와 Process 관계

다시 정리:

```text
Docker Container
≠ Virtual Machine

Container
=
격리된 process environment
```

이다.

Container 안의 ROS 2 node도 결국 host kernel이 scheduling하는 process다.

---

# 92. Mini Practice 1

Docker 설치 환경에서:

```bash
docker version
```

```bash
docker info
```

확인한다.

---

# 93. Mini Practice 2

간단한 Ubuntu container:

```bash
docker run -it ubuntu:22.04 bash
```

안에서:

```bash
uname -m
```

```bash
cat /etc/os-release
```

를 확인한다.

Host와 비교해본다.

---

# 94. Mini Practice 3

Bind mount:

```bash
mkdir -p ~/docker_test
echo hello > ~/docker_test/test.txt
```

실행:

```bash
docker run -it \
  -v ~/docker_test:/data \
  ubuntu:22.04 bash
```

Container에서:

```bash
cat /data/test.txt
```

확인.

---

# 95. Mini Practice 4

Network:

```bash
docker run -it --network host ubuntu:22.04 bash
```

Host와 container의 network interface를 비교한다.

---

# 96. Mini Practice 5

Jetson이라면 container에서:

```bash
uname -m
```

확인.

예:

```text
aarch64
```

GPU-enabled image에서는 PyTorch나 CUDA를 통해 GPU access를 확인한다.

---

# 97. 오늘의 핵심

Docker를 이해할 때 다음 구조가 중요하다.

```text
Application
ROS 2 / CUDA / TensorRT
        │
        ▼
Container
        │
        ▼
Docker Runtime
        │
        ▼
Host Linux Kernel
        │
        ▼
JetPack / Driver
        │
        ▼
Jetson Hardware
```

---

# 98. 반드시 구분할 것

```text
Image ≠ Container

Container ≠ VM

Container OS
≠
Independent Kernel

Docker Network
≠
Host Network

Bind Mount ≠ Image Copy

linux/amd64
≠
linux/arm64

CUDA Container
≠
GPU Driver

Container Running
≠
Application Healthy
```

---

# 99. Jetson Docker Mental Model

최종적으로 Jetson에서:

```text
                Jetson

┌──────────────────────────────┐
│ Hardware                     │
│ ARM CPU / NVIDIA GPU         │
├──────────────────────────────┤
│ JetPack / Linux / Driver     │
├──────────────────────────────┤
│ Docker Runtime               │
├──────────────────────────────┤
│ Container                    │
│                              │
│ ROS 2                        │
│ FAST-LIO2                    │
│ PyTorch                      │
│ TensorRT                     │
└──────────────────────────────┘
```

로 생각하면 된다.

---

# 100. 지금까지 Chapter 연결

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
Docker on Jetson
```

이제 마지막으로 매우 실무적인 내용을 추가한다면:

```text
Chapter 10
ROS 2 + Jetson Debugging & Deployment
```

이 자연스럽다.

Chapter 10에서는 지금까지 배운 모든 layer를 실제 troubleshooting 순서로 묶는다.

```text
Hardware
↓
Linux
↓
Network
↓
Docker
↓
ROS 2
↓
CUDA
↓
Application
```

즉 "로봇이 안 된다"를 실제로 어떻게 진단하는지를 다룬다.