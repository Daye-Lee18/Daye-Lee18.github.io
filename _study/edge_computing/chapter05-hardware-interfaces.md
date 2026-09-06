---
title: "Chapter 5. Hardware Interfaces — Ethernet, CAN, PCIe, USB"
importance: 6
---

> **Goal:** Jetson과 센서, MCU, 다른 컴퓨터가 실제로 어떤 물리적 인터페이스로 연결되는지 이해한다.
> Ethernet, USB, Serial, CAN, PCIe, M.2, CSI의 역할을 구분하고,
> Vision60 같은 로봇 내부의 data flow를 하드웨어 수준에서 이해하는 것이 목표다.

---

# 1. 센서 데이터는 어떻게 Jetson까지 들어올까?

로봇에서 센서는 데이터를 만들고,
Jetson은 그 데이터를 받아 처리한다.

예를 들어:

```text
LiDAR
Camera
IMU
MCU
Other Computer
      │
      ▼
Physical Interface
      │
      ▼
Jetson Hardware
      │
      ▼
Linux Driver
      │
      ▼
Application / ROS 2
```

여기서 Physical Interface에는 여러 종류가 있다.

대표적으로:

```text
Ethernet
USB
Serial
CAN
PCIe
CSI
```

가 있다.

각각 속도, 거리, 안정성, 연결 구조, 사용 목적이 다르다.

---

# 2. Interface란?

Interface는 두 장치가 데이터를 주고받기 위한 **연결 방식 또는 접점**이다.

예를 들어:

```text
LiDAR ── Ethernet ── Jetson
Camera ── USB ── Jetson
MCU ── CAN ── Jetson
SSD ── PCIe ── Jetson
```

각각 서로 다른 interface를 사용한다.

즉:

> Interface = 데이터를 주고받는 물리적/논리적 연결 방식

이라고 생각하면 된다.

---

# 3. Ethernet

Ethernet은 유선 network 통신 기술이다.

많은 사람들이 Ethernet을 단순히:

```text
인터넷 연결용 케이블
```

이라고 생각하지만, 더 정확히는:

> 같은 network 안에서 장치끼리 데이터를 주고받는 통신 방식

이다.

인터넷이 없어도 Ethernet 통신은 가능하다.

예:

```text
Jetson
192.168.0.18
    │
 Ethernet
    │
LiDAR
192.168.0.201
```

이 두 장치가 같은 network에 있다면
인터넷 없이도 직접 데이터를 주고받을 수 있다.

---

# 4. Ethernet과 Internet은 다르다

이 차이는 매우 중요하다.

```text
Ethernet
= Local Network Communication Technology

Internet
= 전 세계 여러 network를 연결한 네트워크
```

예를 들어 로봇 내부:

```text
Jetson
  │
Ethernet
  │
LiDAR
```

만 있어도 통신 가능하다.

여기에는 인터넷이 필요 없다.

---

# 5. IP Address

Ethernet network에서 장치를 구분할 때 IP address를 사용한다.

예:

```text
Jetson
192.168.0.18

LiDAR
192.168.0.201
```

IP address는 network 상에서 장치를 찾기 위한 주소다.

대략:

```text
Device
   │
   └── IP Address
```

라고 생각하면 된다.

---

# 6. MAC Address

MAC address는 network interface 자체에 연결되는 hardware-level identifier다.

예:

```text
00:1A:2B:3C:4D:5E
```

IP address와 MAC address는 역할이 다르다.

| IP Address | MAC Address |
|---|---|
| Network 상의 logical address | Network interface의 hardware address |
| 변경 가능 | 보통 interface에 고유하게 연결 |
| Router가 다루는 주소 | Local network에서 frame 전달에 사용 |

단순히:

```text
IP = 네트워크 주소
MAC = 네트워크 카드 주소
```

정도로 이해하면 된다.

---

# 7. Subnet

두 장치가 같은 local network에 있는지 판단할 때 subnet이 중요하다.

예:

```text
Jetson
192.168.0.18

LiDAR
192.168.0.201

Subnet Mask
255.255.255.0
```

이 경우 일반적으로:

```text
192.168.0.xxx
```

영역을 같은 subnet으로 본다.

그래서 두 장치가 직접 통신하기 쉽다.

---

# 8. Gateway

Gateway는 다른 network로 나갈 때 거쳐 가는 장치다.

예:

```text
Jetson
  │
  ▼
Gateway / Router
  │
  ▼
Company Network
  │
  ▼
Internet
```

로봇 내부 network만 사용할 때는 gateway가 필요 없을 수도 있다.

---

# 9. `ip addr`

Linux에서 network interface와 IP를 확인할 때:

```bash
ip addr
```

를 사용한다.

예:

```text
eth0
wlan0
lo
```

같은 interface를 볼 수 있다.

대표적으로:

```text
eth0
→ Ethernet

wlan0
→ Wi-Fi

lo
→ Loopback
```

이다.

---

# 10. `ping`

두 장치가 IP network 수준에서 통신 가능한지 확인할 때:

```bash
ping 192.168.0.201
```

를 사용한다.

예:

```text
Jetson
   │
   │ ping
   ▼
LiDAR
```

응답이 오면 적어도 network 경로가 존재한다는 의미다.

하지만 `ping`이 된다고 application protocol까지 정상이라는 뜻은 아니다.

---

# 11. Port

하나의 IP address에서 여러 application이 동시에 통신해야 한다.

이때 port 번호를 사용한다.

예:

```text
192.168.0.18:22
```

여기서:

```text
192.168.0.18
→ Device

22
→ SSH service port
```

이다.

예:

```text
SSH → 22
HTTP → 80
HTTPS → 443
```

---

# 12. Ethernet에서 실제 데이터 흐름

예를 들어 LiDAR가 UDP packet을 보낸다고 하자.

```text
LiDAR
  │
  │ Ethernet Frame
  ▼
Network Interface
  │
  ▼
Linux Kernel
  │
  ▼
UDP Socket
  │
  ▼
LiDAR Driver
  │
  ▼
ROS 2 Topic
```

즉 Ethernet cable만 연결된다고 바로 ROS 2 topic이 생기는 것이 아니다.

중간에:

```text
Network Driver
Protocol
Socket
Sensor Driver
```

가 필요하다.

---

# 13. USB

USB는:

**Universal Serial Bus**

의 약자다.

매우 다양한 device를 연결할 수 있다.

예:

```text
Keyboard
Camera
LiDAR
USB Serial Adapter
SSD
Wi-Fi Dongle
```

---

# 14. USB의 특징

USB는 보통:

```text
Host
   │
   └── Device
```

구조다.

예:

```text
Jetson
  │
 USB
  │
Camera
```

여기서 Jetson이 host이고 camera가 device다.

Host가 device를 인식하고 driver를 통해 통신한다.

---

# 15. USB Version

USB에도 여러 version이 있다.

예:

```text
USB 2.0
USB 3.x
USB4
```

version에 따라 최대 bandwidth가 다르다.

Camera나 LiDAR처럼 data rate가 큰 sensor는
USB bandwidth가 충분한지 확인해야 한다.

---

# 16. USB Connector와 USB Protocol은 다르다

중요한 점:

```text
USB-C
```

는 connector 모양이다.

반면:

```text
USB 3.2
USB4
```

는 protocol/version이다.

따라서:

> USB-C라고 해서 항상 같은 속도는 아니다.

USB-C connector를 사용해도 USB 2.0 speed일 수 있다.

---

# 17. Linux에서 USB Device 확인

USB 장치 목록:

```bash
lsusb
```

예:

```text
Bus 001 Device 003: ID ...
```

장치를 연결하기 전/후에 `lsusb`를 비교하면
Linux가 장치를 인식했는지 확인할 수 있다.

---

# 18. USB Serial

많은 MCU나 sensor는 USB를 통해 연결되지만
Linux에서는 serial device처럼 보인다.

예:

```text
/dev/ttyUSB0
/dev/ttyACM0
```

구조:

```text
MCU
  │
 USB
  │
USB-to-Serial
  │
  ▼
Linux
/dev/ttyUSB0
```

---

# 19. Serial Communication

Serial은 데이터를 bit 단위로 순차적으로 전송하는 방식이다.

Embedded system에서 매우 오래전부터 많이 사용된다.

대표적으로:

```text
UART
RS-232
RS-485
```

등이 있다.

---

# 20. UART

UART는:

**Universal Asynchronous Receiver/Transmitter**

이다.

보통:

```text
TX
RX
GND
```

선을 사용한다.

```text
Device A TX ───── Device B RX
Device A RX ───── Device B TX
GND         ───── GND
```

구조다.

---

# 21. Baud Rate

Serial 통신에서는 baud rate를 자주 본다.

예:

```text
9600
115200
921600
```

대략 초당 전송되는 symbol rate를 나타낸다.

실무에서는 두 장치의 baud rate가 같아야 한다.

```text
Device A
115200

Device B
115200
```

처럼 설정해야 정상 통신할 수 있다.

---

# 22. RS-232와 RS-485

둘 다 serial communication 계열이지만 전기적 방식이 다르다.

## RS-232

보통 짧은 거리의 point-to-point 연결에 사용된다.

```text
Device A
   │
   │ RS-232
   ▼
Device B
```

---

## RS-485

차동 신호를 사용해서 noise에 강하고
더 긴 거리에서 사용할 수 있다.

산업 현장에서 많이 사용된다.

```text
Controller
   │
   ├── Device A
   ├── Device B
   └── Device C
```

multi-drop 구조도 가능하다.

---

# 23. CAN Bus

CAN은:

**Controller Area Network**

이다.

자동차와 로봇에서 매우 많이 사용된다.

특히:

```text
Motor Controller
Battery
MCU
Sensor
ECU
```

같은 장치를 연결할 때 사용한다.

---

# 24. 왜 CAN을 로봇에서 많이 사용할까?

로봇 내부는 전기적 noise가 많다.

특히:

```text
Motor
High Current
Power Electronics
```

가 존재한다.

CAN은 이런 환경에서 비교적 robust하게 동작하도록 설계되었다.

장점:

```text
Noise robustness
Multi-node communication
Error detection
Priority-based message arbitration
```

등이 있다.

---

# 25. CAN Bus 구조

CAN은 여러 node가 하나의 bus를 공유할 수 있다.

```text
             CAN Bus

MCU A ──────┬──────── MCU B
            │
            ├──────── Motor Controller
            │
            └──────── Battery Controller
```

일반적으로:

```text
CAN_H
CAN_L
```

두 신호선을 사용한다.

---

# 26. CAN Message는 IP처럼 장치 주소를 쓰나?

CAN에서는 일반적인 Ethernet처럼:

```text
Device IP
```

개념을 사용하지 않는다.

대신 message에 identifier가 있다.

예:

```text
CAN ID 0x101
CAN ID 0x205
```

각 node가 특정 CAN ID의 message를 수신하고 해석한다.

즉:

```text
Ethernet
→ 장치 주소 중심

CAN
→ Message ID 중심
```

이라고 단순화해서 생각할 수 있다.

---

# 27. CAN Arbitration

CAN에서는 여러 node가 동시에 message를 보내려고 할 수 있다.

이때 CAN ID priority를 이용해 arbitration이 이루어진다.

일반적으로 낮은 numerical ID가 더 높은 priority를 가진다.

예:

```text
0x100
0x200
```

가 동시에 전송을 시작하면:

```text
0x100
```

이 우선권을 얻을 수 있다.

---

# 28. Linux에서 CAN

Linux는 CAN interface를 network interface처럼 다룰 수 있다.

SocketCAN을 사용하는 경우:

```text
can0
can1
```

같은 interface를 볼 수 있다.

확인:

```bash
ip link
```

예:

```text
can0
```

---

# 29. CAN Device 올리기

예를 들어:

```bash
sudo ip link set can0 up type can bitrate 500000
```

처럼 CAN interface를 설정할 수 있다.

여기서:

```text
bitrate 500000
```

은:

```text
500 kbps
```

를 의미한다.

정확한 bitrate는 bus에 연결된 모든 device 설정과 맞아야 한다.

---

# 30. CAN Message 확인

SocketCAN tool이 설치되어 있다면:

```bash
candump can0
```

으로 CAN message를 볼 수 있다.

보낼 때는:

```bash
cansend
```

같은 tool을 사용할 수 있다.

---

# 31. CAN과 ROS 2

구조는 보통:

```text
Motor Controller
      │
      │ CAN
      ▼
Linux can0
      │
      ▼
CAN Driver Node
      │
      ▼
ROS 2 Topic
```

이다.

즉 ROS 2가 CAN 자체를 대체하는 것은 아니다.

---

# 32. PCIe

PCIe는:

**Peripheral Component Interconnect Express**

이다.

컴퓨터 내부에서 고속 device를 연결하기 위한 interface다.

예:

```text
GPU
NVMe SSD
Network Card
Accelerator
Capture Card
```

등에 사용된다.

---

# 33. PCIe는 왜 빠른가?

PCIe는 high-speed point-to-point link를 사용한다.

대략:

```text
CPU / SoC
    │
   PCIe
    │
Device
```

형태다.

Bandwidth가 매우 높아서 대용량 data transfer에 적합하다.

---

# 34. PCIe Lane

PCIe에서는:

```text
x1
x2
x4
x8
x16
```

같은 표현을 본다.

여기서 `x4`는 4개의 lane을 사용한다는 의미다.

일반적으로 lane 수가 많을수록 더 높은 bandwidth를 제공할 수 있다.

---

# 35. PCIe Generation

또:

```text
PCIe Gen3
PCIe Gen4
PCIe Gen5
```

같은 표현을 본다.

새 generation일수록 lane당 bandwidth가 증가한다.

따라서 실제 bandwidth는:

```text
Generation
×
Lane Count
```

에 영향을 받는다.

---

# 36. M.2

M.2는 매우 헷갈리는 개념이다.

M.2는 기본적으로 **form factor / connector specification**이다.

즉:

```text
M.2 = 물리적 형태
```

에 가깝다.

M.2 slot을 통해 여러 protocol을 사용할 수 있다.

예:

```text
PCIe
SATA
USB
```

등이다.

---

# 37. M.2 NVMe SSD

Jetson에 NVMe SSD를 연결할 때 보통:

```text
M.2
+
PCIe
+
NVMe
```

가 함께 등장한다.

구조:

```text
Jetson
  │
  │ PCIe
  ▼
M.2 Connector
  │
  ▼
NVMe SSD
```

여기서:

```text
M.2 = connector/form factor
PCIe = transport interface
NVMe = storage protocol
```

이라고 구분해야 한다.

---

# 38. SATA와 NVMe

Storage에서 자주 비교한다.

```text
SATA SSD
NVMe SSD
```

NVMe SSD는 일반적으로 PCIe를 이용해 더 높은 bandwidth를 제공한다.

로봇에서:

```text
rosbag
Video
Point Cloud
```

처럼 대용량 데이터를 저장한다면 storage throughput이 중요하다.

---

# 39. CSI Camera

Jetson에서는 camera 연결에 CSI를 많이 사용한다.

CSI는 일반적으로:

**Camera Serial Interface**

를 의미한다.

MIPI CSI 계열을 사용한다.

구조:

```text
Camera Sensor
     │
     │ MIPI CSI
     ▼
Jetson SoC
```

---

# 40. USB Camera vs CSI Camera

## USB Camera

```text
Camera
  │
 USB
  │
Jetson
```

장점:

```text
쉽게 연결 가능
범용성 높음
```

---

## CSI Camera

```text
Camera Sensor
   │
 MIPI CSI
   │
Jetson
```

장점:

```text
Embedded system에 적합
낮은 overhead 가능
Jetson ISP pipeline 활용 가능
```

---

# 41. ISP란?

ISP는:

**Image Signal Processor**

이다.

Raw camera sensor data를 실제 image로 처리한다.

예:

```text
Raw Bayer Image
     │
     ▼
ISP
     │
     ├── Demosaicing
     ├── White Balance
     ├── Noise Reduction
     └── Color Processing
     │
     ▼
RGB Image
```

Jetson에는 camera processing에 특화된 hardware block이 존재할 수 있다.

---

# 42. Ethernet LiDAR

많은 3D LiDAR는 Ethernet을 사용한다.

예:

```text
LiDAR
  │
  │ UDP packets
  ▼
Ethernet
  │
  ▼
Jetson
  │
  ▼
LiDAR Driver
  │
  ▼
PointCloud2
```

대용량 point cloud를 지속적으로 보내야 하기 때문에 Ethernet이 적합하다.

---

# 43. UDP란?

Ethernet 위에서는 여러 transport protocol을 사용할 수 있다.

대표적으로:

```text
TCP
UDP
```

가 있다.

LiDAR sensor는 UDP를 사용하는 경우가 많다.

UDP는 connection-oriented reliability보다
낮은 overhead와 빠른 전달을 중요하게 여긴다.

---

# 44. TCP vs UDP

단순 비교:

| TCP | UDP |
|---|---|
| 연결 지향 | 비연결형 |
| 순서/재전송 지원 | 기본적으로 재전송 없음 |
| 신뢰성 중심 | 낮은 latency/overhead |
| Web, SSH 등 | Streaming, sensor data 등에 자주 사용 |

하지만 실제 protocol 선택은 application 요구에 따라 달라진다.

---

# 45. Packet이란?

Network에서는 데이터를 한 번에 거대한 덩어리로 보내는 것이 아니라
작은 단위로 나누어 전달한다.

이를 흔히 packet이라고 부른다.

예:

```text
LiDAR Scan
     │
     ▼
Packet 1
Packet 2
Packet 3
...
     │
     ▼
Jetson
     │
     ▼
Driver reconstructs point cloud
```

---

# 46. Driver란?

Driver는 hardware와 operating system/application 사이를 연결하는 software다.

예:

```text
LiDAR Hardware
      │
      ▼
Network / USB
      │
      ▼
Driver
      │
      ▼
Application
```

ROS에서는 sensor vendor driver가 ROS node 형태로 제공되는 경우가 많다.

예:

```text
Velodyne
   │
   ▼
velodyne_driver
   │
   ▼
ROS 2 Topic
```

---

# 47. Hardware Driver와 ROS Driver

엄밀히는 여러 layer가 있을 수 있다.

```text
Sensor
  │
  ▼
Linux Kernel Driver
  │
  ▼
User-space SDK
  │
  ▼
ROS 2 Driver Node
  │
  ▼
ROS Topic
```

어떤 sensor는 kernel driver가 필요하고,
어떤 sensor는 Ethernet socket만 사용해서 user-space에서 직접 통신할 수도 있다.

---

# 48. Vision60 내부 연결을 생각해보자

단순화하면 다음과 같은 구조를 상상할 수 있다.

```text
                     Vision60

LiDAR ─── Ethernet ──────┐
                         │
Camera ─── USB / CSI ────┤
                         ▼
                   Jetson / Xavier
                         │
                         ├── ROS 2
                         ├── SLAM
                         ├── Perception
                         │
                         ▼
                    MCU / Controller
                         │
                      CAN / Other Bus
                         │
                         ▼
                    Joint Motors
```

실제 Vision60의 정확한 내부 연결은 hardware revision과 configuration에 따라 다를 수 있지만,
개념적으로는 이 구조로 이해할 수 있다.

---

# 49. Xavier와 Orin을 Ethernet으로 연결한다는 의미

예를 들어 로봇 안에:

```text
Xavier
Orin
```

두 대의 computer가 있다고 하자.

Ethernet으로 연결하면:

```text
Xavier
192.168.1.10
      │
   Ethernet
      │
Orin
192.168.1.20
```

두 컴퓨터는 하나의 local network에서 통신할 수 있다.

그러면:

```text
SSH
ROS 2
TCP
UDP
File Transfer
```

등이 가능하다.

---

# 50. ROS 2는 Physical Interface가 아니다

이 부분은 매우 중요하다.

ROS 2는:

```text
Ethernet
CAN
USB
```

같은 physical interface가 아니다.

ROS 2는 application-level robotics middleware/framework다.

예:

```text
ROS 2
   │
   ▼
DDS
   │
   ▼
UDP / Network
   │
   ▼
Ethernet / Wi-Fi
```

같은 구조가 될 수 있다.

즉:

```text
ROS 2
≠ Ethernet

ROS 2
≠ TCP

ROS 2
≠ CAN
```

이다.

---

# 51. Network Stack

Ethernet communication을 layer로 보면:

```text
ROS 2 Application
       │
       ▼
DDS / Middleware
       │
       ▼
UDP / TCP
       │
       ▼
IP
       │
       ▼
Ethernet
       │
       ▼
Cable / PHY
```

각 layer가 서로 다른 역할을 한다.

---

# 52. PHY란?

Ethernet documentation에서:

```text
PHY
```

라는 단어를 볼 수 있다.

PHY는 physical layer transceiver다.

Digital network data를 실제 cable signal로 바꾸고,
반대로 cable signal을 digital data로 바꾼다.

```text
SoC
 │
MAC
 │
PHY
 │
Ethernet Cable
```

---

# 53. Bandwidth

Bandwidth는 일정 시간 동안 얼마나 많은 데이터를 전송할 수 있는지를 의미한다.

예:

```text
100 Mbps
1 Gbps
10 Gbps
```

LiDAR, camera처럼 data rate가 큰 sensor를 여러 개 연결하면
bandwidth가 부족할 수 있다.

---

# 54. Latency

Latency는 데이터가 한 지점에서 다른 지점까지 전달되는 데 걸리는 시간이다.

예:

```text
Sensor
   │
   │ 5 ms
   ▼
Jetson
```

로봇에서는 bandwidth뿐 아니라 latency도 매우 중요하다.

---

# 55. Bandwidth와 Latency는 다르다

비유하면:

```text
Bandwidth
= 도로 차선 수

Latency
= 목적지까지 걸리는 시간
```

차선이 많다고 반드시 목적지까지 빨리 도착하는 것은 아니다.

---

# 56. Sensor Interface를 고를 때 보는 것

어떤 interface가 좋은지는 application에 따라 다르다.

확인할 요소:

```text
Data Rate
Latency
Cable Length
Noise
Reliability
Power
Connector
Cost
Number of Devices
```

---

# 57. 간단 비교

| Interface | 주요 용도 |
|---|---|
| Ethernet | LiDAR, computer-to-computer, network device |
| USB | Camera, serial adapter, general peripheral |
| UART | MCU, simple sensor |
| RS-485 | Industrial serial communication |
| CAN | Motor controller, automotive/robot bus |
| PCIe | NVMe, accelerator, high-speed device |
| MIPI CSI | Embedded camera |

---

# 58. 문제를 디버깅하는 순서

Sensor 데이터가 안 들어올 때 다음 순서로 보면 좋다.

```text
Physical Connection
      │
      ▼
Interface Detection
      │
      ▼
Linux Device / Network
      │
      ▼
Driver
      │
      ▼
ROS 2 Node
      │
      ▼
ROS 2 Topic
      │
      ▼
Application
```

예를 들어 Ethernet LiDAR라면:

```text
Cable connected?
      ↓
Link up?
      ↓
IP correct?
      ↓
ping?
      ↓
UDP packets arriving?
      ↓
LiDAR driver running?
      ↓
ROS topic published?
      ↓
FAST-LIO2 subscribed?
```

순서로 확인할 수 있다.

---

# 59. USB Sensor Debugging

USB sensor라면:

```text
Device connected?
      ↓
lsusb
      ↓
/dev device exists?
      ↓
Permission?
      ↓
Driver running?
      ↓
ROS topic?
```

으로 볼 수 있다.

---

# 60. CAN Debugging

CAN이라면:

```text
Physical wiring?
      ↓
CAN_H / CAN_L?
      ↓
Termination?
      ↓
Bitrate?
      ↓
can0 up?
      ↓
candump can0
      ↓
Driver node?
```

순서로 확인한다.

---

# 61. CAN Termination Resistor

CAN bus에서는 일반적으로 bus 양 끝에 termination resistor가 필요하다.

대표적으로:

```text
120 Ω
```

resistor를 사용한다.

구조:

```text
120Ω                           120Ω
 │                              │
 ▼                              ▼
Node A ===== CAN BUS ===== Node B
```

termination이 잘못되면 signal reflection 때문에 통신 문제가 발생할 수 있다.

---

# 62. Full Duplex와 Half Duplex

통신에서는:

```text
Full Duplex
Half Duplex
```

라는 표현도 본다.

## Full Duplex

송신과 수신을 동시에 가능.

```text
A → B
A ← B
```

동시 가능.

---

## Half Duplex

한 시점에는 한 방향 통신 중심.

```text
A → B

or

A ← B
```

---

# 63. Ethernet Switch

여러 Ethernet device를 연결할 때 switch를 사용한다.

예:

```text
             Switch
           /   |    \
          /    |     \
      Jetson LiDAR  Xavier
```

Switch는 local Ethernet network에서 frame을 적절한 port로 전달한다.

---

# 64. Router와 Switch 차이

단순화하면:

```text
Switch
→ 같은 local network 안의 device 연결

Router
→ 서로 다른 network 연결
```

예:

```text
Robot Internal Network
        │
      Router
        │
Company Network
```

---

# 65. Robot 내부 Network

로봇 내부 network는 인터넷과 독립적으로 구성할 수 있다.

예:

```text
Robot Network

Xavier      192.168.10.10
Orin        192.168.10.11
LiDAR       192.168.10.20
Camera      192.168.10.30
```

이 장치들은 서로 통신할 수 있다.

인터넷 연결은 별도의 interface를 통해 제공할 수 있다.

---

# 66. 두 개의 Network Interface

Jetson이 다음처럼 두 network에 동시에 연결될 수도 있다.

```text
Jetson

eth0
│
└── Robot Network
    192.168.10.x

wlan0
│
└── Company Wi-Fi
    10.x.x.x
```

이 경우 Jetson은:

```text
Robot Internal Communication
+
Company Network / Internet
```

을 동시에 사용할 수 있다.

단, routing과 security 설정을 제대로 해야 한다.

---

# 67. Routing

Routing은 packet을 어느 network interface로 보낼지 결정하는 과정이다.

확인:

```bash
ip route
```

예:

```text
default via ...
192.168.10.0/24 dev eth0
```

같은 정보가 나온다.

---

# 68. Vision60에서 네트워크 구조를 볼 때

예를 들어:

```text
Mac
 │
Company Wi-Fi
 │
Router
 │
Orin
 │
Ethernet
 │
Xavier
 │
Robot Internal Network
```

처럼 구성할 수 있다.

이 경우 중요한 질문은:

```text
Mac에서 Xavier까지 route가 있는가?
Firewall은 허용하는가?
IP subnet이 맞는가?
```

이다.

---

# 69. ROS 2와 Network

ROS 2가 여러 computer에 걸쳐 동작할 때 network가 중요하다.

예:

```text
Xavier
ROS 2 Node A
      │
      │ Ethernet
      ▼
Orin
ROS 2 Node B
```

둘이 같은 ROS 2 domain과 compatible middleware/network 설정을 사용하면
topic을 주고받을 수 있다.

---

# 70. ROS_DOMAIN_ID

ROS 2에서:

```bash
export ROS_DOMAIN_ID=123
```

같은 설정을 사용할 수 있다.

같은 network에 있어도 domain이 다르면 discovery가 분리될 수 있다.

예:

```text
Xavier
ROS_DOMAIN_ID=123

Orin
ROS_DOMAIN_ID=123
```

이면 같은 ROS domain에 들어갈 수 있다.

---

# 71. DDS

ROS 2는 내부 middleware로 DDS 계열을 사용하는 경우가 많다.

구조:

```text
ROS 2 Node
    │
    ▼
rclcpp / rclpy
    │
    ▼
RMW
    │
    ▼
DDS
    │
    ▼
UDP / Network
```

이 구조는 다음 Chapter에서 더 자세히 다룬다.

---

# 72. 오늘의 핵심

센서와 Jetson 사이에는 항상 실제 data path가 존재한다.

```text
Sensor
   │
   ▼
Physical Interface
   │
   ▼
Linux
   │
   ▼
Driver
   │
   ▼
ROS 2
   │
   ▼
Algorithm
```

그리고 다음을 구분해야 한다.

```text
Ethernet ≠ Internet

USB-C ≠ USB Speed

M.2 ≠ PCIe

PCIe ≠ NVMe

ROS 2 ≠ Ethernet

CAN ≠ ROS 2

IP Address ≠ MAC Address
```

---

# 73. Robot Hardware Mental Model

Vision60 같은 로봇에서:

```text
                  Robot

LiDAR ── Ethernet ──────┐
                        │
Camera ── USB / CSI ────┤
                        ▼
                    Jetson
                        │
                        ├── Linux
                        ├── ROS 2
                        ├── SLAM
                        └── AI
                        │
                        ▼
                     MCU
                        │
                     CAN Bus
                        │
                        ▼
                 Motor Controllers
```

처럼 생각할 수 있다.

---

# 74. Mini Practice

Jetson에서 network 확인:

```bash
ip addr
```

```bash
ip route
```

```bash
ping <device-ip>
```

---

USB 확인:

```bash
lsusb
```

---

Serial device 확인:

```bash
ls /dev/ttyUSB*
```

```bash
ls /dev/ttyACM*
```

---

PCIe device 확인:

```bash
lspci
```

---

Storage 확인:

```bash
lsblk
```

---

Network interface 확인:

```bash
ip link
```

---

CAN interface가 있다면:

```bash
ip link show can0
```

---

# Next Chapter

## Chapter 6. ROS 2 as a Robotics Middleware

다음 Chapter에서는 지금까지의 hardware와 Linux 위에서
ROS 2가 정확히 어떤 역할을 하는지 살펴본다.

- ROS 2는 Operating System인가?
- ROS 2는 Protocol인가?
- ROS 2를 Framework라고 불러도 되는가?
- Node란 무엇인가?
- Topic, Service, Action은 무엇이 다른가?
- Publisher와 Subscriber는 무엇인가?
- `rclcpp`, `rclpy`는 무엇인가?
- RMW는 무엇인가?
- DDS는 무엇인가?
- CycloneDDS와 Fast DDS는 무엇이 다른가?
- TCP/UDP와 ROS 2는 어떤 관계인가?
- Network가 없어도 ROS 2 communication이 가능한가?
- ROS_DOMAIN_ID는 왜 필요한가?
- QoS는 무엇인가?
- SensorDataQoS는 왜 LiDAR/IMU에서 사용하는가?

Chapter 6에서는 지금까지 배운:

```text
Hardware
+
Linux
+
Network
```

위에 ROS 2가 어떤 layer로 올라가는지 연결해서 이해한다.