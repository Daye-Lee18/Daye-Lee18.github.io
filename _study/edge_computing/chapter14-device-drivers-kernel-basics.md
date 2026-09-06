---
title: "Chapter 14. Device Drivers & Kernel Basics"
importance: 15
---

> **Goal:** Linux에서 hardware가 software와 어떻게 연결되는지 이해한다.
>
> Kernel, User Space, Kernel Space, Device Driver, Kernel Module, `/dev`, udev,
> Interrupt, DMA, System Call의 기본 개념을 이해하고,
> 센서 데이터가 hardware에서 ROS 2 node까지 올라오는 경로를 시스템 관점에서 이해한다.

---

# 1. 센서를 꽂으면 왜 바로 프로그램에서 읽을 수 있을까?

예를 들어 USB IMU를 Jetson에 연결했다고 하자.

```text
IMU
 │
 USB
 │
 ▼
Jetson
```

우리는 Linux에서:

```bash
ls /dev/ttyUSB0
```

같은 device를 볼 수 있다.

그리고 프로그램에서는:

```text
/dev/ttyUSB0
```

를 열어서 데이터를 읽을 수 있다.

하지만 실제로는 중간에 많은 layer가 있다.

```text
Sensor Hardware
      │
      ▼
Physical Interface
      │
      ▼
Linux Kernel
      │
      ▼
Device Driver
      │
      ▼
Device File
      │
      ▼
User Program
      │
      ▼
ROS 2 Node
```

---

# 2. Kernel 다시 보기

Kernel은 Operating System의 핵심이다.

Application이 hardware를 직접 제어하지 않고,
Kernel을 통해 resource를 사용한다.

```text
Application
    │
    ▼
Linux Kernel
    │
    ├── CPU
    ├── Memory
    ├── Storage
    ├── Network
    ├── USB
    └── Devices
```

Kernel은 hardware와 software 사이의 핵심 중간 계층이다.

---

# 3. User Space와 Kernel Space

Linux에서는 실행 영역을 크게:

```text
User Space
Kernel Space
```

로 나누어 생각할 수 있다.

---

# 4. User Space

일반 application이 실행되는 영역이다.

예:

```text
ROS 2 Node
FAST-LIO2
Python
C++ Application
Docker Process
Sensor SDK
```

대부분의 우리가 작성하는 프로그램은 user space에서 실행된다.

---

# 5. Kernel Space

Kernel과 핵심 device driver 등이 실행되는 privileged 영역이다.

```text
Kernel Space

├── Scheduler
├── Memory Management
├── Network Stack
├── Filesystem
└── Device Drivers
```

hardware에 더 직접적으로 접근한다.

---

# 6. 왜 User Space와 Kernel Space를 나눌까?

일반 application이 hardware와 memory를 마음대로 제어하면 위험하다.

예:

```text
Bad Application
      ↓
잘못된 memory access
      ↓
전체 system crash
```

가능성이 커진다.

그래서:

```text
Application
      │
      ▼
Controlled Interface
      │
      ▼
Kernel
```

구조를 사용한다.

---

# 7. Privilege

Kernel은 높은 privilege를 가진다.

User application은 제한된 privilege로 실행된다.

따라서 application이:

```text
File
Network
Device
Memory
```

를 사용하려면 kernel이 제공하는 interface를 사용한다.

---

# 8. System Call

User-space program이 kernel 기능을 요청하는 방법 중 핵심이:

```text
System Call
```

이다.

예:

```text
open()
read()
write()
close()
```

등.

---

# 9. File 읽기 예

C program이 file을 읽는다고 하자.

```text
Application
    │
    │ read()
    ▼
System Call
    │
    ▼
Kernel
    │
    ▼
Filesystem
    │
    ▼
Storage
```

이다.

---

# 10. Device도 비슷하다

예:

```text
/dev/ttyUSB0
```

에서 data를 읽는 경우:

```text
Application
    │
    │ read()
    ▼
Kernel
    │
    ▼
USB / Serial Driver
    │
    ▼
Hardware
```

형태다.

---

# 11. Driver란?

Device Driver는:

> Operating System이 특정 hardware와 통신할 수 있도록 해주는 software

이다.

예:

```text
USB Driver
Network Driver
Camera Driver
Serial Driver
GPU Driver
Storage Driver
```

---

# 12. 왜 Driver가 필요할까?

각 hardware는 제어 방법이 다르다.

예:

```text
Camera
LiDAR
Network Card
NVMe SSD
```

는 서로 전혀 다른 register, protocol, timing을 사용할 수 있다.

Application이 이 모든 hardware 세부사항을 직접 알 필요 없도록
driver가 중간에서 추상화한다.

---

# 13. Driver의 역할

```text
Hardware-specific details
        ↓
Driver
        ↓
Standard OS interface
```

예를 들어 camera는:

```text
/dev/video0
```

처럼 보일 수 있다.

Serial device는:

```text
/dev/ttyUSB0
```

처럼 보일 수 있다.

---

# 14. Linux Device File

Linux에서는 많은 hardware를:

```text
File-like interface
```

로 표현한다.

그래서:

```text
Everything is a file
```

이라는 표현을 자주 듣는다.

정확히 모든 것이 진짜 regular file인 것은 아니지만,
많은 resource를 file descriptor interface로 다룬다는 의미다.

---

# 15. `/dev`

Device file은 보통:

```text
/dev
```

아래에 나타난다.

예:

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/video0
/dev/nvme0n1
```

---

# 16. Character Device

Device file에는 여러 종류가 있다.

대표적으로:

```text
Character Device
Block Device
```

가 있다.

Character device는 byte stream 중심으로 데이터를 처리한다.

예:

```text
Serial Port
Terminal
Some sensors
```

---

# 17. Block Device

Block 단위 storage device다.

예:

```text
SSD
HDD
NVMe
```

Linux에서는:

```text
/dev/sda
/dev/nvme0n1
```

같은 형태로 보일 수 있다.

---

# 18. `ls -l /dev`

예:

```bash
ls -l /dev/ttyUSB0
```

출력 앞부분:

```text
crw-rw----
```

여기서 첫 문자:

```text
c
```

는 character device를 의미할 수 있다.

Block device는:

```text
b
```

로 표시될 수 있다.

---

# 19. Major / Minor Number

Device file에는:

```text
Major Number
Minor Number
```

가 있다.

개념적으로:

```text
Major
→ 어떤 driver인가?

Minor
→ 그 driver가 관리하는 어떤 device인가?
```

를 구분한다.

---

# 20. Kernel Module

Linux kernel 기능 일부는 필요할 때 동적으로 load할 수 있다.

이를:

```text
Kernel Module
```

이라고 한다.

예:

```text
USB driver
CAN driver
Filesystem driver
```

일부가 module 형태일 수 있다.

---

# 21. `.ko`

Kernel module file은 흔히:

```text
.ko
```

확장자를 사용한다.

`ko`는:

```text
Kernel Object
```

를 의미한다.

예:

```text
my_driver.ko
```

---

# 22. Loaded Module 확인

```bash
lsmod
```

현재 load된 kernel module 목록을 볼 수 있다.

---

# 23. Module 정보

```bash
modinfo <module>
```

예:

```bash
modinfo can
```

module metadata를 확인할 수 있다.

---

# 24. Module Load

```bash
sudo modprobe <module>
```

를 사용할 수 있다.

예:

```bash
sudo modprobe can
```

---

# 25. `insmod`와 `modprobe`

둘 다 kernel module을 load할 수 있다.

`insmod`:

```text
특정 .ko file 직접 load
```

`modprobe`:

```text
module dependency까지 고려하여 load
```

그래서 일반적으로 `modprobe`를 더 많이 사용한다.

---

# 26. Module Remove

```bash
sudo modprobe -r <module>
```

또는:

```bash
sudo rmmod <module>
```

를 사용할 수 있다.

실제 robot에서는 module 제거가 hardware 동작에 영향을 줄 수 있으므로 주의해야 한다.

---

# 27. Built-in Driver

모든 driver가 module은 아니다.

일부 driver는 kernel image 자체에 포함되어 있다.

즉:

```text
Driver
├── Built-in
└── Loadable Module
```

두 형태가 가능하다.

---

# 28. Hardware가 연결되면 어떻게 Driver를 찾을까?

예를 들어 USB device를 연결한다.

```text
USB Device
    │
    ▼
USB Controller
    │
    ▼
Kernel detects device
    │
    ▼
Vendor / Product ID 확인
    │
    ▼
Matching Driver
```

형태로 동작할 수 있다.

---

# 29. USB VID / PID

USB device에는:

```text
Vendor ID
Product ID
```

가 있다.

`lsusb`에서:

```text
1234:5678
```

같은 값을 볼 수 있다.

```text
1234
→ Vendor ID

5678
→ Product ID
```

이다.

---

# 30. `lsusb`

USB device 확인:

```bash
lsusb
```

장치를 연결하기 전후를 비교하면
kernel이 USB device를 인식했는지 볼 수 있다.

---

# 31. Kernel Log 확인

Device 연결 시:

```bash
dmesg
```

를 확인할 수 있다.

예:

```bash
dmesg | tail
```

USB device가 연결되면:

```text
new USB device found
converter now attached to ttyUSB0
```

같은 message가 나타날 수 있다.

---

# 32. `dmesg`

`dmesg`는 kernel ring buffer message를 보여준다.

Hardware debugging에서 매우 중요하다.

예:

```text
USB disconnect
Network link down
Driver error
I/O error
```

등.

---

# 33. udev

Device가 연결되면 `/dev` node 생성과 naming 등을 관리하는 중요한 user-space system이:

```text
udev
```

이다.

---

# 34. udev 역할

예:

```text
USB Serial Device 연결
      │
      ▼
Kernel detects
      │
      ▼
udev
      │
      ▼
/dev/ttyUSB0
```

같은 흐름을 생각할 수 있다.

---

# 35. 왜 `ttyUSB0`가 바뀔 수 있을까?

두 USB serial device가 있다고 하자.

Boot 순서에 따라:

```text
Sensor A → ttyUSB0
Sensor B → ttyUSB1
```

일 수도 있고,

다음 boot:

```text
Sensor B → ttyUSB0
Sensor A → ttyUSB1
```

이 될 수도 있다.

---

# 36. Robot에서는 큰 문제다

Config에:

```text
IMU = /dev/ttyUSB0
```

라고 hard-code했는데
boot 후 다른 device가 ttyUSB0가 되면 잘못된 sensor를 열 수 있다.

---

# 37. udev Rule

이 문제를 해결하기 위해 udev rule을 만들 수 있다.

예:

```text
특정 VID/PID/Serial Number
        ↓
/dev/vision60_imu
```

같은 stable device name을 만들 수 있다.

---

# 38. udev Rule 위치

보통:

```text
/etc/udev/rules.d/
```

에 custom rule을 둘 수 있다.

예:

```text
99-robot-sensors.rules
```

---

# 39. Stable Device Naming

예:

```text
/dev/vision60_imu
/dev/robot_lidar
```

처럼 이름을 사용하면:

```text
ttyUSB0
ttyUSB1
```

보다 config가 안정적이다.

---

# 40. Permission도 udev로 설정할 수 있다

Device가:

```text
root:dialout
```

owner/group을 갖도록 하거나
특정 permission을 설정할 수 있다.

하지만 너무 넓게:

```text
MODE="0777"
```

같이 주는 것은 security 측면에서 좋지 않을 수 있다.

---

# 41. Group Permission

Serial device에서 자주 보는 group:

```text
dialout
```

사용자가 이 group에 속해 있으면
sudo 없이 serial device에 접근할 수 있는 경우가 많다.

확인:

```bash
groups
```

---

# 42. User를 Group에 추가

예:

```bash
sudo usermod -aG dialout $USER
```

변경 후 새 login session이 필요할 수 있다.

실제 permission policy는 system 환경에 맞춰야 한다.

---

# 43. Network Device Driver

Ethernet adapter도 driver가 필요하다.

구조:

```text
Ethernet PHY / NIC
       │
       ▼
Kernel Network Driver
       │
       ▼
eth0
       │
       ▼
Linux Network Stack
```

---

# 44. `eth0`도 `/dev` file인가?

일반적으로 network interface는:

```text
/dev/eth0
```

같은 device file로 다루지 않는다.

Linux network subsystem에서:

```text
eth0
wlan0
```

같은 network interface로 표현된다.

즉 모든 hardware가 동일한 `/dev` interface를 사용하는 것은 아니다.

---

# 45. Network Stack

```text
Application
    │
 Socket
    │
 TCP / UDP
    │
 IP
    │
 Network Driver
    │
 NIC
    │
 Ethernet
```

구조다.

---

# 46. Socket

Network application이 kernel network stack을 사용하는 interface다.

예:

```text
LiDAR Driver
    │
    │ UDP Socket
    ▼
Linux Kernel
    │
    ▼
Ethernet Driver
```

---

# 47. User-Space Driver란?

"Driver"라는 단어는 조금 넓게 사용된다.

ROS에서:

```text
velodyne_driver
```

같은 것을 driver라고 부르지만,
이것이 반드시 kernel driver라는 뜻은 아니다.

---

# 48. Kernel Driver vs User-Space Driver

예:

```text
Ethernet LiDAR
```

의 경우:

```text
LiDAR Hardware
      │
      ▼
Ethernet NIC
      │
      ▼
Kernel NIC Driver
      │
      ▼
UDP Socket
      │
      ▼
User-Space LiDAR Driver
      │
      ▼
ROS 2 Topic
```

이다.

---

# 49. ROS Driver

ROS 2 driver node는 보통:

```text
Hardware / SDK / Socket data
           ↓
      ROS Message
```

로 변환한다.

예:

```text
Raw LiDAR packets
       ↓
ROS Driver
       ↓
sensor_msgs/PointCloud2
```

---

# 50. 따라서 Driver는 여러 Layer에 있다

```text
Hardware
   │
   ▼
Kernel Driver
   │
   ▼
User-Space SDK
   │
   ▼
ROS Driver Node
   │
   ▼
ROS Topic
```

어디를 말하는지 문맥을 확인해야 한다.

---

# 51. Camera Example

USB camera:

```text
Camera
  │
 USB
  │
 ▼
USB Controller
  │
 ▼
Linux Kernel Driver
  │
 ▼
V4L2
  │
 ▼
/dev/video0
  │
 ▼
ROS Camera Node
  │
 ▼
sensor_msgs/Image
```

---

# 52. V4L2

V4L2:

```text
Video4Linux2
```

Linux video capture/device framework다.

Camera program에서 자주 사용된다.

---

# 53. Serial IMU Example

```text
IMU
 │
USB/UART
 │
 ▼
Kernel Serial Driver
 │
 ▼
/dev/ttyUSB0
 │
 ▼
IMU SDK / ROS Driver
 │
 ▼
sensor_msgs/Imu
```

---

# 54. CAN Example

```text
CAN Controller
      │
      ▼
CAN Kernel Driver
      │
      ▼
SocketCAN
      │
      ▼
can0
      │
      ▼
User Application
      │
      ▼
ROS 2 Node
```

---

# 55. SocketCAN

Linux는 CAN을 network socket과 비슷한 interface로 다루는:

```text
SocketCAN
```

framework를 제공한다.

그래서:

```bash
ip link show can0
```

처럼 network command로 CAN을 확인할 수 있다.

---

# 56. GPU Driver

Jetson의 NVIDIA GPU도 kernel-level driver와 user-space library가 함께 작동한다.

개념적으로:

```text
CUDA Application
      │
      ▼
CUDA Runtime
      │
      ▼
NVIDIA User-Space Driver
      │
      ▼
Kernel Driver
      │
      ▼
GPU
```

이다.

---

# 57. Container와 Driver

Docker container가 host kernel을 공유한다는 것은
driver 관점에서도 중요하다.

```text
Container
CUDA App
    │
    ▼
Host Kernel Driver
    │
    ▼
Jetson GPU
```

그래서 container 안에 완전히 별도의 kernel GPU driver를 실행하는 구조가 아니다.

---

# 58. Interrupt

Hardware는 CPU에게:

> "처리할 일이 생겼다."

고 알려야 할 때가 있다.

이때 사용하는 중요한 mechanism이:

```text
Interrupt
```

이다.

---

# 59. Polling

Interrupt와 비교되는 방식:

```text
Polling
```

CPU가 계속:

```text
Data 왔어?
Data 왔어?
Data 왔어?
```

라고 확인하는 방식이다.

---

# 60. Polling Example

```text
CPU
 │
 ├── check device
 ├── check device
 ├── check device
 ├── check device
 └── ...
```

Device에 아무 data가 없어도 CPU time을 사용할 수 있다.

---

# 61. Interrupt Example

```text
CPU
 │
 └── 다른 작업 수행

Device
 │
 └── data ready
       ↓
    Interrupt
       ↓
CPU handles event
```

필요할 때 hardware가 CPU에게 알린다.

---

# 62. Interrupt 장점

불필요한 polling을 줄일 수 있다.

하지만 interrupt가 너무 자주 발생하면:

```text
Interrupt Overhead
```

도 생길 수 있다.

---

# 63. Interrupt Handler

Interrupt가 발생하면 kernel은 해당 interrupt를 처리하는 code를 실행한다.

이를 일반적으로:

```text
Interrupt Handler
```

라고 한다.

---

# 64. Interrupt Context

Interrupt handler는 일반 application context와 다르다.

빠르게 처리하고 긴 작업은 다른 mechanism으로 넘기는 설계가 일반적이다.

Kernel 내부에서는 매우 중요한 개념이다.

---

# 65. IRQ

Linux에서 interrupt를:

```text
IRQ
```

라고 표현하는 것을 자주 볼 수 있다.

IRQ:

```text
Interrupt Request
```

이다.

---

# 66. Interrupt 확인

```bash
cat /proc/interrupts
```

를 보면 CPU별 interrupt count를 볼 수 있다.

예:

```text
NIC
USB
Timer
```

등의 interrupt가 어느 CPU에서 처리되는지 볼 수 있다.

---

# 67. 왜 Robot Performance에서 중요할까?

고속 network/camera/sensor device가 많으면
interrupt load가 증가할 수 있다.

특정 CPU core가 interrupt 처리에 너무 많은 시간을 쓰면
application performance에도 영향을 줄 수 있다.

---

# 68. DMA

DMA:

```text
Direct Memory Access
```

이다.

Hardware가 CPU를 거치지 않고 memory와 직접 데이터를 전송할 수 있도록 하는 mechanism이다.

---

# 69. DMA가 없다면

아주 단순화하면:

```text
Device
   ↓
CPU
   ↓
Memory
```

CPU가 데이터 이동에 계속 관여해야 할 수 있다.

---

# 70. DMA가 있으면

```text
Device
   │
   │ DMA
   ▼
Memory

CPU
→ setup / completion handling
```

형태로 CPU 개입을 줄일 수 있다.

---

# 71. 왜 DMA가 중요한가?

Camera, network, storage처럼 data rate가 높은 device에서는
모든 byte를 CPU가 직접 복사하면 부담이 크다.

DMA를 이용하면 CPU는:

```text
Data Movement
```

보다:

```text
Actual Computation
```

에 더 많은 시간을 쓸 수 있다.

---

# 72. Camera + DMA

개념적으로:

```text
Camera
   │
   ▼
Hardware Interface
   │
   ▼
DMA
   │
   ▼
RAM Buffer
   │
   ▼
Application
```

구조가 가능하다.

---

# 73. NIC + DMA

Network card도 packet data를 RAM buffer로 이동할 때
DMA를 사용할 수 있다.

```text
Ethernet
   ↓
NIC
   ↓
DMA
   ↓
RAM
   ↓
Kernel Network Stack
```

---

# 74. NVMe + DMA

고속 NVMe storage도 DMA를 활용해
storage와 system memory 사이 data transfer를 효율적으로 처리한다.

---

# 75. DMA = Zero Copy?

아니다.

```text
DMA
≠
Zero Copy
```

DMA는:

```text
누가 memory transfer를 수행하는가?
```

에 관한 개념이다.

Zero-copy는:

```text
불필요한 buffer copy를 얼마나 줄이는가?
```

에 관한 개념이다.

---

# 76. Memory-Mapped I/O

CPU가 device register를 memory address처럼 접근하는 방식도 있다.

이를:

```text
Memory-Mapped I/O
```

라고 한다.

개념:

```text
CPU Address Space

RAM
...
Device Register
...
```

처럼 특정 address가 hardware register와 연결된다.

---

# 77. Register

Hardware device 내부의 작은 control/status storage다.

예:

```text
Start
Stop
Status
Buffer Address
Interrupt Enable
```

등을 register로 제어할 수 있다.

Driver가 이 register를 읽고 쓰며 hardware를 제어한다.

---

# 78. Application이 Register를 직접 만질까?

일반적으로는 아니다.

구조:

```text
Application
   ↓
Driver API
   ↓
Kernel Driver
   ↓
Hardware Register
```

를 사용한다.

직접 hardware register에 접근하면 portability와 security 문제가 크다.

---

# 79. System Call 비용

User space에서 kernel space로 넘어가는 데는 일정 overhead가 있다.

예:

```text
User
 ↓ system call
Kernel
 ↓
User
```

그래서 고성능 I/O에서는 system call 수를 줄이는 최적화가 중요할 수 있다.

---

# 80. Buffering

I/O에서 매 byte마다 system call을 하면 비효율적일 수 있다.

그래서:

```text
Buffer
```

를 사용한다.

예:

```text
Data
Data
Data
Data
   ↓
Buffer
   ↓
One Larger Operation
```

---

# 81. Blocking I/O

`read()`를 호출했는데 data가 아직 없다면
process가 기다릴 수 있다.

이를:

```text
Blocking I/O
```

라고 한다.

---

# 82. Non-Blocking I/O

Data가 없으면 바로 반환하고
application이 다른 작업을 할 수 있다.

```text
Non-Blocking I/O
```

라고 한다.

---

# 83. `select`, `poll`, `epoll`

Linux에서는 여러 file descriptor의 I/O event를 효율적으로 기다리기 위한 mechanism이 있다.

예:

```text
select
poll
epoll
```

Network server나 high-performance application에서 자주 사용된다.

---

# 84. ROS에서는 이런 것을 직접 안 보는데?

ROS user는 보통:

```cpp
subscription callback
```

만 작성한다.

하지만 아래에서는:

```text
DDS
Socket
Kernel
Interrupt
Driver
```

등이 동작하고 있다.

즉 ROS 2가 hardware layer를 없애는 것은 아니다.

---

# 85. Abstraction

상위 software는 아래 복잡한 detail을 숨긴다.

예:

```text
FAST-LIO2

subscribe("/imu")
```

만 보면 되지만 아래에서는:

```text
ROS
DDS
Socket
Network Stack
NIC Driver
DMA
Ethernet
```

이 동작한다.

---

# 86. Abstraction Layer

전체 stack:

```text
Algorithm
   │
   ▼
ROS 2
   │
   ▼
User-Space Driver / SDK
   │
   ▼
System Calls / Sockets
   │
   ▼
Linux Kernel
   │
   ▼
Kernel Driver
   │
   ▼
Bus / Interface
   │
   ▼
Hardware
```

---

# 87. Kernel Version

Driver compatibility에서 kernel version이 중요할 수 있다.

확인:

```bash
uname -r
```

예:

```text
5.x.x-...
```

---

# 88. 왜 Kernel Version이 중요할까?

Driver는 kernel API와 연결된다.

Kernel version이 바뀌면
외부 driver module이 다시 build되어야 할 수도 있다.

---

# 89. Out-of-Tree Driver

Linux kernel source tree 밖에서 별도로 제공되는 driver를:

```text
Out-of-Tree Driver
```

라고 부른다.

Vendor hardware에서 볼 수 있다.

---

# 90. Kernel Update 위험

Robot에서 무심코 kernel을 업데이트하면:

```text
Custom Driver
GPU Driver
CAN Driver
Sensor Driver
```

가 깨질 가능성이 있다.

따라서 embedded/robot production에서는 OS/kernel update를 신중하게 관리한다.

---

# 91. DKMS

DKMS:

```text
Dynamic Kernel Module Support
```

이다.

Kernel이 업데이트될 때 external kernel module을 자동으로 rebuild하는 데 사용될 수 있다.

---

# 92. Jetson에서는 더 조심해야 한다

Jetson은:

```text
JetPack
Jetson Linux
Kernel
NVIDIA Driver
CUDA
```

가 밀접하게 연결되어 있다.

그래서 일반 Ubuntu PC처럼 아무 package나 무작정 upgrade하면
compatibility 문제가 발생할 수 있다.

---

# 93. Device Tree

Embedded Linux에서 자주 등장하는 개념:

```text
Device Tree
```

이다.

Hardware가 어떤 peripheral과 연결되어 있는지 kernel에 설명한다.

---

# 94. 왜 Device Tree가 필요할까?

PC는 hardware discovery가 상대적으로 dynamic한 경우가 많지만,
embedded board에서는:

```text
UART
I2C
SPI
GPIO
Camera
```

가 board wiring에 고정되어 있을 수 있다.

Kernel이 이 hardware 구성을 알아야 한다.

---

# 95. Device Tree Example 개념

```text
UART Controller
Status = okay

I2C Sensor
Address = 0x68
```

같은 hardware 정보를 표현할 수 있다.

실제 syntax는 더 복잡하다.

---

# 96. Device Tree Blob

Device Tree source:

```text
.dts
```

compile 후:

```text
.dtb
```

형태를 사용할 수 있다.

---

# 97. GPIO

GPIO:

```text
General Purpose Input/Output
```

이다.

단순 digital signal을 읽거나 출력할 수 있다.

예:

```text
Button
LED
Trigger
Reset
```

---

# 98. I2C

I2C는 embedded sensor 연결에 자주 사용하는 bus다.

일반적으로:

```text
SDA
SCL
```

두 신호선을 사용한다.

여러 device가 같은 bus를 공유할 수 있다.

---

# 99. I2C Address

각 I2C device는 address를 가진다.

예:

```text
0x68
```

Linux에서:

```bash
i2cdetect
```

같은 tool을 사용할 수 있는 환경도 있다.

---

# 100. SPI

SPI도 embedded device communication에 자주 사용된다.

대표 신호:

```text
MOSI
MISO
SCLK
CS
```

I2C보다 더 높은 speed가 필요한 경우 사용할 수 있다.

---

# 101. UART / I2C / SPI

단순 비교:

| Interface | 특징 |
|---|---|
| UART | 간단한 point-to-point serial |
| I2C | 여러 low-speed device 공유 가능 |
| SPI | 높은 speed, chip select 필요 |
| CAN | robust multi-node robot/automotive bus |
| Ethernet | 고속 network communication |

---

# 102. Driver Debugging 순서

Sensor가 안 잡히면:

```text
1. Power?
   ↓
2. Physical connection?
   ↓
3. Bus detects device?
   ↓
4. Kernel detects device?
   ↓
5. Driver loaded?
   ↓
6. Device node/interface exists?
   ↓
7. Permission?
   ↓
8. User-space SDK?
   ↓
9. ROS driver?
   ↓
10. ROS topic?
```

순서로 본다.

---

# 103. USB Sensor Debugging

```text
Sensor connected?
   ↓
lsusb
   ↓
dmesg
   ↓
driver loaded?
   ↓
/dev/ttyUSB0?
   ↓
permission?
   ↓
ROS node?
```

---

# 104. Ethernet LiDAR Debugging

Ethernet LiDAR는 조금 다르다.

```text
LiDAR powered?
   ↓
NIC link?
   ↓
Kernel network driver?
   ↓
eth0?
   ↓
IP?
   ↓
ping?
   ↓
UDP packet?
   ↓
User-space LiDAR driver?
   ↓
ROS topic?
```

---

# 105. CAN Debugging

```text
CAN hardware?
   ↓
Kernel CAN driver?
   ↓
can0?
   ↓
bitrate?
   ↓
candump?
   ↓
User node?
   ↓
ROS?
```

---

# 106. Kernel Panic

Kernel 자체에서 심각한 문제가 발생하면:

```text
Kernel Panic
```

이 발생할 수 있다.

일반 user application crash보다 훨씬 심각하다.

---

# 107. Segmentation Fault와 Kernel Panic 차이

```text
Segmentation Fault
→ 주로 특정 user process가 crash

Kernel Panic
→ Kernel이 더 이상 안전하게 실행될 수 없는 상태
```

이다.

---

# 108. Segfault가 나도 OS는 살아 있을 수 있다

예:

```text
FAST-LIO2 segfault
```

이면 FAST-LIO2 process만 죽고
Linux 자체는 계속 동작할 수 있다.

이것이 user/kernel isolation의 장점 중 하나다.

---

# 109. Driver Bug는 더 위험할 수 있다

Kernel driver bug는 kernel space에서 발생하기 때문에
전체 system 안정성에 영향을 줄 가능성이 더 크다.

---

# 110. Kernel Log가 중요한 이유

Sensor가 사라지거나 network adapter가 reset될 때
application log보다 kernel log가 더 많은 정보를 줄 수 있다.

확인:

```bash
dmesg -T
```

또는:

```bash
journalctl -k
```

---

# 111. `journalctl -k`

Kernel 관련 journal message:

```bash
journalctl -k
```

Boot 기준:

```bash
journalctl -k -b
```

등을 사용할 수 있다.

---

# 112. Hardware Error와 Software Error 구분

예:

```text
ROS node says:
Cannot open /dev/ttyUSB0
```

이것만 보면 ROS 문제처럼 보이지만 실제 원인은:

```text
USB disconnected
Driver not loaded
Permission denied
udev naming changed
```

일 수 있다.

---

# 113. Bottom-Up Debugging

그래서 hardware 문제는 항상 아래에서 위로 본다.

```text
Hardware
   ↓
Kernel
   ↓
Device
   ↓
Driver
   ↓
User Space
   ↓
ROS
   ↓
Application
```

---

# 114. Top-Down Debugging의 문제

바로 FAST-LIO2 code부터 보는 경우:

```text
Application
   ↓
???
   ↓
Sensor
```

원인이 hardware layer에 있다면 많은 시간을 낭비한다.

---

# 115. DMA와 ROS 2 연결

예를 들어 camera:

```text
Camera
   ↓
DMA
   ↓
RAM Buffer
   ↓
Camera Driver
   ↓
ROS Image
   ↓
AI
```

이다.

DMA는 ROS 2 기능은 아니지만
ROS 2가 받는 data가 효율적으로 올라오는 데 중요한 underlying mechanism이다.

---

# 116. Interrupt와 ROS Callback

둘도 같은 개념이 아니다.

```text
Hardware Interrupt
→ Kernel-level hardware event

ROS Callback
→ User-space application event
```

이다.

하지만 전체 chain에서는 연결될 수 있다.

---

# 117. 전체 Event Flow

예:

```text
IMU Data Ready
     │
     ▼
Hardware Interrupt
     │
     ▼
Kernel Driver
     │
     ▼
Buffer
     │
     ▼
User-Space Driver
     │
     ▼
ROS Publisher
     │
     ▼
DDS
     │
     ▼
FAST-LIO2 Subscriber
     │
     ▼
imu_cbk()
```

---

# 118. 이것이 왜 중요한가?

`imu_cbk()`가 늦게 들어온다고 하자.

원인은 callback code만이 아닐 수 있다.

```text
Sensor timing
Interrupt
Driver buffering
USB latency
ROS queue
DDS
Executor
CPU load
```

어디에서든 latency가 생길 수 있다.

---

# 119. Performance Stack

```text
Hardware
   │
Interrupt / DMA
   │
Kernel Driver
   │
System Call
   │
User Driver
   │
ROS 2
   │
Application
```

어느 layer가 bottleneck인지 profiling해야 한다.

---

# 120. Real-Time Computing과 연결

다음 Chapter 15에서는:

```text
Interrupt latency
Scheduling latency
Priority
Jitter
PREEMPT_RT
```

등을 더 자세히 다룬다.

Device driver와 kernel을 알아야 real-time computing도 이해할 수 있다.

---

# 121. Mini Practice 1

Jetson에서:

```bash
uname -r
```

실행한다.

질문:

```text
현재 kernel version은?
```

---

# 122. Mini Practice 2

```bash
lsmod | head
```

현재 load된 kernel module 일부를 본다.

---

# 123. Mini Practice 3

USB device를 연결한 뒤:

```bash
lsusb
```

그리고:

```bash
dmesg | tail -n 30
```

을 확인한다.

질문:

```text
Kernel이 어떤 device로 인식했는가?
```

---

# 124. Mini Practice 4

Serial device가 있다면:

```bash
ls -l /dev/ttyUSB*
```

확인.

다음 정보 확인:

```text
Character device?
Owner?
Group?
Permission?
```

---

# 125. Mini Practice 5

```bash
cat /proc/interrupts
```

을 확인한다.

질문:

```text
Network / USB 관련 interrupt가 있는가?

어떤 CPU에서 count가 증가하는가?
```

---

# 126. Mini Practice 6

Network interface:

```bash
ip link
```

확인.

그다음:

```bash
ethtool -i eth0
```

지원된다면 어떤 network driver가 사용되는지 확인한다.

---

# 127. Mini Practice 7

Storage:

```bash
lsblk
```

그리고:

```bash
udevadm info --query=all --name=/dev/nvme0n1
```

같은 방식으로 udev 정보를 확인할 수 있다.

실제 device path는 시스템에 맞춰 바꾼다.

---

# 128. Mini Practice 8

ROS sensor 하나를 골라 data path를 직접 작성한다.

예:

```text
LiDAR Hardware
→ ?
→ ?
→ ROS Driver
→ PointCloud2
→ FAST-LIO2
```

`?` 부분에 실제 Linux/network/driver layer를 채운다.

---

# 129. 반드시 구분할 것

```text
Kernel
≠
Ubuntu

User Space
≠
Kernel Space

Kernel Driver
≠
ROS Driver

Device File
≠
Regular File

Character Device
≠
Block Device

Kernel Module
≠
Kernel 전체

Interrupt
≠
ROS Callback

Polling
≠
Interrupt

DMA
≠
Zero-Copy

eth0
≠
/dev/eth0

Driver
≠
Application
```

---

# 130. Device Data Mental Model

Sensor data가 올라오는 전체 흐름:

```text
Hardware
   │
   ▼
Electrical / Physical Interface
   │
   ▼
Controller
   │
   ▼
Kernel Driver
   │
   ▼
Interrupt / DMA
   │
   ▼
Kernel Buffer
   │
   ▼
System Call / Socket
   │
   ▼
User-Space Driver
   │
   ▼
ROS 2 Node
   │
   ▼
ROS Message
   │
   ▼
Robot Algorithm
```

---

# 131. Vision60 Mental Model

예를 들어 Vision60 LiDAR:

```text
LiDAR
   │
Ethernet
   │
   ▼
Jetson NIC
   │
   ▼
NIC Kernel Driver
   │
   ▼
Linux Network Stack
   │
   ▼
UDP Socket
   │
   ▼
LiDAR ROS Driver
   │
   ▼
PointCloud2
   │
   ▼
FAST-LIO2
```

IMU:

```text
IMU
   │
USB / Serial / Other Interface
   │
   ▼
Kernel Driver
   │
   ▼
Device / Buffer
   │
   ▼
IMU ROS Driver
   │
   ▼
sensor_msgs/Imu
   │
   ▼
FAST-LIO2
```

---

# 132. Chapter 14 핵심

Robot software가 sensor를 읽을 수 있는 이유는
ROS 2가 hardware를 직접 제어하기 때문이 아니다.

실제로는:

```text
Hardware
↓
Kernel
↓
Driver
↓
User Space
↓
ROS 2
↓
Algorithm
```

이라는 여러 layer가 존재한다.

이 구조를 이해하면:

```text
"센서가 안 들어온다."
```

라는 문제를:

```text
Hardware?
Kernel?
Driver?
Permission?
ROS?
Application?
```

으로 분리해서 생각할 수 있다.

---

# Next Chapter

## Chapter 15. Real-Time Computing

다음 Chapter에서는 로봇에서 매우 중요한:

```text
Real-Time
Latency
Jitter
Deadline
Scheduler
Priority
Context Switch
Interrupt Latency
PREEMPT_RT
CPU Affinity
Priority Inversion
Watchdog
```

을 다룬다.

특히:

```text
"100 Hz로 동작한다"
```

와:

```text
"매 10 ms마다 반드시 실행된다"
```

가 왜 전혀 다른 의미인지 설명한다.

그리고:

```text
Jetson
→ Perception / SLAM / Navigation

MCU / Real-Time Controller
→ Motor Control
```

처럼 왜 로봇에서 compute 역할을 나누는지도 이해하게 된다.