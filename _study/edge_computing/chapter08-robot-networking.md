---
title: "Chapter 8. Networking for Robots"
importance: 9
---

> **Goal:** 로봇 내부 네트워크와 외부 네트워크의 차이를 이해하고,
> IP, subnet, gateway, routing, DNS, DHCP, SSH, Wi-Fi, Ethernet, ROS 2 통신이
> 실제 Vision60/Xavier/Orin 환경에서 어떻게 연결되는지 이해한다.

---

# 1. 로봇에서 Network가 왜 중요한가?

현대 로봇은 하나의 컴퓨터만 사용하는 경우가 드물다.

예를 들어:

```text
Vision60
├── Xavier
├── Orin
├── LiDAR
├── Camera
├── MCU
├── Operator Laptop
└── Company Network
```

이 장치들이 서로 데이터를 주고받으려면 network 구조를 이해해야 한다.

예:

```text
MacBook
   │
   ▼
Company Wi-Fi
   │
   ▼
Jetson Orin
   │
   ▼
Ethernet
   │
   ▼
Xavier
   │
   ▼
Robot Internal Network
```

---

# 2. Network와 Internet은 다르다

이 차이는 반드시 기억해야 한다.

```text
Network
= 장치들이 서로 연결되어 통신하는 구조

Internet
= 전 세계의 여러 network를 연결한 거대한 network
```

따라서:

```text
Internet 없어도
Robot 내부 Network는 가능
```

하다.

예:

```text
Jetson
192.168.10.10

LiDAR
192.168.10.20
```

두 장치가 Ethernet으로 연결되어 있으면
인터넷 없이도 통신할 수 있다.

---

# 3. LAN

LAN은:

**Local Area Network**

이다.

가까운 범위의 장치들이 연결된 network를 의미한다.

예:

```text
Robot LAN
```

```text
Office LAN
```

예를 들어 Vision60 내부:

```text
Xavier
Orin
LiDAR
```

가 하나의 LAN에 있을 수 있다.

---

# 4. IP Address

IP address는 network에서 장치를 식별하기 위한 logical address다.

예:

```text
192.168.0.18
```

구조적으로는:

```text
Device
   │
   └── IP Address
```

라고 생각하면 된다.

---

# 5. IPv4

우리가 흔히 보는:

```text
192.168.0.18
10.0.0.5
172.16.0.3
```

같은 주소는 IPv4다.

IPv4는 32-bit address다.

```text
192 . 168 . 0 . 18
```

네 개의 8-bit 구간으로 표현된다.

---

# 6. Private IP

로컬 네트워크에서 자주 사용하는 private IP 범위가 있다.

대표적으로:

```text
10.0.0.0/8

172.16.0.0/12

192.168.0.0/16
```

그래서 로봇 내부에서:

```text
192.168.x.x
```

나:

```text
10.x.x.x
```

를 자주 보게 된다.

---

# 7. Public IP와 Private IP

Private IP는 로컬 network에서 사용한다.

Public IP는 인터넷에서 식별 가능한 address다.

예:

```text
Robot Device
192.168.0.18
      │
      ▼
Router
      │
      ▼
Public IP
      │
      ▼
Internet
```

---

# 8. Subnet

IP만 보는 것보다 subnet을 이해하는 것이 중요하다.

예:

```text
IP:
192.168.0.18

Subnet Mask:
255.255.255.0
```

이 경우 일반적으로:

```text
192.168.0.x
```

범위를 같은 local subnet으로 본다.

CIDR notation으로는:

```text
192.168.0.18/24
```

처럼 표현할 수 있다.

---

# 9. `/24`는 무슨 뜻인가?

```text
/24
```

는 앞쪽 24 bit가 network 부분이라는 뜻이다.

예:

```text
192.168.0.18/24
```

이면:

```text
Network:
192.168.0.0

Host range:
192.168.0.x
```

형태다.

---

# 10. 같은 Subnet이면 무엇이 좋은가?

예:

```text
Xavier
192.168.10.10/24

Orin
192.168.10.20/24
```

둘은 같은 subnet에 있으므로
보통 router 없이 직접 통신할 수 있다.

```text
Xavier
   │
Ethernet
   │
Orin
```

---

# 11. 다른 Subnet이면?

예:

```text
Xavier
192.168.10.10/24

MacBook
10.0.0.20/24
```

이 둘은 서로 다른 subnet이다.

따라서 직접 통신하려면:

```text
Router
Routing
Gateway
```

등이 필요할 수 있다.

---

# 12. Default Gateway

Gateway는 다른 network로 나가기 위한 출구 역할을 한다.

예:

```text
Jetson
192.168.0.18
      │
      ▼
Gateway
192.168.0.1
      │
      ▼
Internet / Other Network
```

default gateway는:

> 목적지가 local subnet이 아닐 때 packet을 어디로 보낼지

알려주는 기본 경로다.

---

# 13. Routing

Routing은:

> packet을 어느 network interface와 gateway로 보낼지 결정하는 과정

이다.

Linux에서 확인:

```bash
ip route
```

예:

```text
default via 192.168.0.1 dev wlan0
192.168.10.0/24 dev eth0
```

이 의미는:

```text
192.168.10.x
→ eth0로 보냄

그 외 기본 목적지
→ wlan0를 통해 gateway로 보냄
```

이다.

---

# 14. Jetson에 Ethernet과 Wi-Fi가 동시에 있을 수 있다

예:

```text
Jetson

eth0
192.168.10.10
→ Robot Internal Network

wlan0
10.20.30.40
→ Company Wi-Fi
```

이런 구성은 매우 흔하다.

```text
LiDAR
   │
   ▼
eth0
Jetson
wlan0
   │
   ▼
Company Network
```

---

# 15. 이 구조의 장점

로봇 내부 통신과 인터넷/회사망을 분리할 수 있다.

예:

```text
eth0
→ LiDAR / Xavier / MCU gateway

wlan0
→ SSH / Git / Internet / Company Services
```

이렇게 하면 sensor traffic과 외부 traffic을 분리할 수 있다.

---

# 16. Routing을 잘못 설정하면 생기는 문제

예를 들어 LiDAR가:

```text
192.168.10.20
```

인데 Jetson이 해당 subnet route를 가지고 있지 않으면:

```text
Jetson
   │
   X
LiDAR
```

통신이 되지 않는다.

또 default route가 이상하게 설정되면
인터넷 traffic이 엉뚱한 interface로 나갈 수도 있다.

---

# 17. `ip addr`

현재 network interface와 IP 확인:

```bash
ip addr
```

예:

```text
eth0
wlan0
lo
```

각 interface에 어떤 IP가 붙어 있는지 본다.

---

# 18. `ip link`

interface 자체의 상태를 확인:

```bash
ip link
```

예:

```text
state UP
state DOWN
```

Ethernet cable이 연결되어 있지 않거나 interface가 disable되어 있으면
통신이 되지 않을 수 있다.

---

# 19. `ip route`

routing table 확인:

```bash
ip route
```

robot network 문제를 볼 때 매우 중요하다.

---

# 20. `ping`

두 장치가 IP 수준에서 통신 가능한지 확인:

```bash
ping 192.168.10.20
```

응답이 오면:

```text
IP reachability
```

가 있다는 뜻이다.

하지만 application이 정상이라는 보장은 아니다.

---

# 21. ping이 된다고 ROS 2가 되는 것은 아니다

예:

```text
ping OK
```

여도 ROS 2 discovery가 안 될 수 있다.

원인:

```text
ROS_DOMAIN_ID mismatch
DDS config
Firewall
Multicast issue
RMW mismatch
```

등이 있을 수 있다.

따라서:

```text
ping OK
≠
ROS 2 OK
```

이다.

---

# 22. MAC Address

Network interface에는 MAC address가 있다.

확인:

```bash
ip link
```

예:

```text
link/ether 00:11:22:33:44:55
```

MAC address는 local Ethernet network에서 frame을 전달하는 데 사용된다.

---

# 23. ARP

IPv4 local network에서는:

```text
IP Address
→ MAC Address
```

를 알아내는 과정이 필요하다.

이때 ARP를 사용한다.

구조:

```text
"192.168.10.20 누구야?"
       │
       ▼
ARP Request
       │
       ▼
Device responds with MAC
```

확인:

```bash
ip neigh
```

---

# 24. Switch

같은 Ethernet LAN 안에서 여러 device를 연결할 때 switch를 사용한다.

```text
          Switch
        /   |    \
       /    |     \
   Xavier  Orin  LiDAR
```

Switch는 MAC address를 보고 Ethernet frame을 전달한다.

---

# 25. Router

Router는 서로 다른 IP network를 연결한다.

예:

```text
Robot Network
192.168.10.0/24
       │
       ▼
     Router
       │
       ▼
Company Network
10.0.0.0/8
```

즉:

```text
Switch
→ 같은 LAN 연결

Router
→ 서로 다른 network 연결
```

---

# 26. DHCP

DHCP는 device에 network configuration을 자동으로 배정하는 protocol이다.

예:

```text
Laptop joins Wi-Fi
      │
      ▼
DHCP Server
      │
      ▼
IP Address
Subnet
Gateway
DNS
```

을 자동으로 받는다.

---

# 27. Static IP

Robot sensor에서는 static IP를 많이 사용한다.

예:

```text
LiDAR
192.168.10.20

Jetson
192.168.10.10
```

주소가 계속 바뀌면 driver configuration이 불편하기 때문이다.

---

# 28. DHCP vs Static IP

| DHCP | Static IP |
|---|---|
| 자동 IP 할당 | 직접 고정 |
| Laptop/Office에서 편리 | Sensor/Robot에서 편리 |
| IP가 바뀔 수 있음 | 주소가 일정 |
| DHCP server 필요 | 직접 관리 필요 |

---

# 29. DNS

DNS는:

**Domain Name System**

이다.

사람이:

```text
github.com
```

을 입력하면 network에서는 실제 IP가 필요하다.

DNS가:

```text
github.com
      ↓
IP Address
```

로 변환한다.

---

# 30. DNS가 안 되면?

예:

```bash
ping 8.8.8.8
```

은 되는데:

```bash
ping github.com
```

이 안 된다면 DNS 문제일 수 있다.

즉:

```text
Internet routing OK
DNS resolution FAIL
```

일 가능성이 있다.

---

# 31. Hostname

각 computer에는 hostname이 있다.

확인:

```bash
hostname
```

예:

```text
vision60-xavier
```

Hostname은 사람이 장치를 쉽게 식별하기 위한 이름이다.

IP address와는 별개다.

---

# 32. `/etc/hosts`

간단한 hostname-to-IP mapping을 직접 설정할 수도 있다.

예:

```text
192.168.10.10 xavier
192.168.10.20 orin
```

그러면:

```bash
ssh xavier
```

처럼 사용할 수도 있다.

단, DNS나 mDNS 같은 다른 방법도 존재한다.

---

# 33. SSH 다시 보기

SSH는:

**Secure Shell**

이다.

예:

```bash
ssh user@192.168.10.10
```

구조:

```text
MacBook
   │
   │ SSH over TCP/IP
   ▼
Jetson
```

SSH는 보통 TCP port 22를 사용한다.

---

# 34. SSH는 어디에서 실행되는가?

Mac에서:

```bash
ssh user@jetson
```

한 뒤:

```bash
ros2 launch ...
```

를 실행하면 실제 ROS 2 process는 Jetson에서 실행된다.

```text
MacBook
   │
   │ keyboard / terminal I/O
   ▼
Jetson
   │
   └── ros2 launch process
```

---

# 35. SCP

SSH 기반으로 파일을 복사할 수 있다.

예:

```bash
scp file.txt user@192.168.10.10:/home/user/
```

반대로 remote에서 local로:

```bash
scp user@192.168.10.10:/home/user/log.txt .
```

---

# 36. rsync

대량의 directory를 동기화할 때 `rsync`를 많이 사용한다.

예:

```bash
rsync -av folder/ user@192.168.10.10:/home/user/folder/
```

변경된 파일 위주로 전송할 수 있어 효율적이다.

---

# 37. `rsync`는 Network가 필요하다

remote SSH 주소를 사용하는:

```bash
rsync user@robot:/path .
```

형태는 해당 robot과 network 통신이 가능해야 한다.

즉 Vision60 Wi-Fi에 연결되어 있거나
회사 network에서 robot까지 route가 있어야 한다.

---

# 38. Wi-Fi

Wi-Fi도 IP network를 전달하는 link technology다.

Ethernet과 비슷하게:

```text
IP
TCP/UDP
ROS 2
SSH
```

를 위에 올릴 수 있다.

차이는 physical/link layer가 wireless라는 것이다.

---

# 39. Ethernet vs Wi-Fi

| Ethernet | Wi-Fi |
|---|---|
| 유선 | 무선 |
| 일반적으로 안정적 | 간섭 영향 가능 |
| 낮은 jitter 가능 | 환경 영향 큼 |
| cable 필요 | 이동성이 좋음 |
| Robot 내부에 적합 | Operator access에 편리 |

---

# 40. 로봇 내부 통신은 Ethernet이 유리한 경우가 많다

LiDAR처럼 대용량 sensor data는:

```text
Bandwidth
Latency
Reliability
```

가 중요하다.

그래서 Ethernet을 사용하는 경우가 많다.

반면 operator laptop 접속은 Wi-Fi가 편리하다.

---

# 41. AP란?

AP는:

**Access Point**

이다.

Wi-Fi 장치들이 무선 network에 접속할 수 있게 한다.

예:

```text
Vision60 Wi-Fi AP
       │
       ├── Remote Controller
       ├── Laptop
       └── Xavier
```

---

# 42. Vision60 자체 Wi-Fi

Robot이 자체 Wi-Fi AP를 만들 수도 있다.

예:

```text
MacBook
   │
   │ Wi-Fi
   ▼
Vision60 AP
   │
   ▼
Xavier
```

이 경우 인터넷이 없어도 robot에 SSH 접속할 수 있다.

---

# 43. 회사 Wi-Fi와 Robot Wi-Fi

예를 들어:

```text
Company Wi-Fi
→ Internet 있음

Vision60 Wi-Fi
→ Robot 내부 통신
→ Internet 없음
```

일 수 있다.

이 둘은 다른 network다.

---

# 44. 두 Network를 연결할 수 있을까?

가능하다.

예:

```text
Company Wi-Fi
      │
      ▼
Orin
      │
   Ethernet
      │
      ▼
Xavier / Robot Network
```

Orin이 두 network 사이에 위치할 수 있다.

하지만 단순히 cable만 연결한다고 routing이 자동으로 되는 것은 아니다.

---

# 45. IP Forwarding

Linux computer를 router처럼 사용하려면
IP forwarding을 사용할 수 있다.

개념:

```text
Company Network
      │
      ▼
Orin
      │
      ▼
Robot Network
```

Orin이 packet을 한 interface에서 다른 interface로 전달한다.

---

# 46. NAT

외부 network에서 내부 private network로 인터넷을 공유할 때
NAT를 사용할 수 있다.

예:

```text
Xavier
192.168.10.10
      │
      ▼
Orin
      │ NAT
      ▼
Company Wi-Fi
      │
      ▼
Internet
```

이 경우 Xavier가 직접 public network에 노출되지 않고도
internet을 사용할 수 있다.

---

# 47. NAT는 무엇을 하는가?

단순화하면 내부 주소를 외부 통신에 맞게 변환한다.

```text
192.168.10.10
      │
      ▼
NAT
      │
      ▼
Orin external IP
```

여러 내부 장치가 하나의 외부 interface를 공유할 수 있다.

---

# 48. NAT와 Routing은 다르다

Routing:

```text
어느 길로 packet을 보낼까?
```

NAT:

```text
packet의 address를 어떻게 변환할까?
```

이다.

---

# 49. Firewall

Linux에는 firewall이 있을 수 있다.

Firewall은:

```text
어떤 traffic을 허용하고
어떤 traffic을 차단할지
```

결정한다.

예:

```text
SSH allowed
ROS multicast blocked
```

같은 상황도 가능하다.

---

# 50. UFW

Ubuntu에서는:

```bash
ufw
```

를 사용할 수 있다.

상태 확인:

```bash
sudo ufw status
```

실제 robot에서는 무작정 firewall을 끄기보다
필요한 traffic만 허용하는 것이 좋다.

---

# 51. Port 다시 보기

한 IP에서 여러 application을 구분하기 위해 port를 사용한다.

예:

```text
192.168.10.10:22
→ SSH

192.168.10.10:8080
→ Web Application
```

---

# 52. TCP

TCP는 connection-oriented transport protocol이다.

특징:

```text
Connection
Reliable Delivery
Ordering
Retransmission
```

SSH는 TCP를 사용한다.

```text
SSH
 ↓
TCP
 ↓
IP
 ↓
Ethernet / Wi-Fi
```

---

# 53. UDP

UDP는 비교적 단순하고 낮은 overhead를 가진 transport protocol이다.

특징:

```text
No guaranteed delivery
No built-in retransmission
Low overhead
```

LiDAR sensor stream이나 DDS에서 UDP를 사용하는 경우가 많다.

---

# 54. TCP와 UDP를 비교하면

```text
TCP
→ 신뢰성 중심

UDP
→ 낮은 overhead / 빠른 전달 중심
```

이라고 단순화할 수 있다.

하지만 application이 요구하는 특성에 따라 선택한다.

---

# 55. Multicast

Multicast는 하나의 sender가 여러 receiver에게 데이터를 전달하는 방식이다.

```text
       Sender
         │
         ▼
     Multicast
      /   |   \
     /    |    \
Node A Node B Node C
```

DDS discovery 등에서 multicast가 사용될 수 있다.

---

# 56. Multicast가 막히면 ROS 2 문제가 생길 수 있다

회사 Wi-Fi나 일부 router에서는 multicast traffic이 제한될 수 있다.

그 결과:

```text
ping은 됨
SSH도 됨
ROS 2 node discovery 안 됨
```

같은 상황이 발생할 수 있다.

---

# 57. ROS 2 Network Mental Model

ROS 2를 multi-computer에서 사용할 때:

```text
ROS 2
   │
   ▼
RMW
   │
   ▼
DDS
   │
   ▼
UDP / Shared Memory / Other Transport
   │
   ▼
IP
   │
   ▼
Ethernet / Wi-Fi
```

형태로 본다.

---

# 58. ROS_DOMAIN_ID

같은 network라고 해서 모든 ROS 2 node가 자동으로 같은 system에 속하는 것은 아니다.

예:

```bash
export ROS_DOMAIN_ID=123
```

Xavier와 Orin의 domain이 같아야 하는 경우가 많다.

---

# 59. RMW Implementation

예:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

두 machine이 완전히 같은 RMW를 써야만 통신 가능한 것은 아니지만,
실무에서는 동일한 middleware configuration으로 맞추는 것이 디버깅을 쉽게 만든다.

---

# 60. CycloneDDS Configuration

ROS 2 system에서:

```text
CYCLONEDDS_URI
```

같은 environment variable을 볼 수 있다.

예:

```bash
export CYCLONEDDS_URI=/path/to/cyclonedds.xml
```

CycloneDDS가 어떤 interface를 사용하고
어떻게 discovery할지 설정할 수 있다.

---

# 61. Network Interface가 여러 개면 DDS가 헷갈릴 수 있다

예:

```text
Jetson

eth0 → Robot Network
wlan0 → Company Wi-Fi
docker0 → Docker Network
```

DDS가 어떤 interface로 discovery traffic을 보내야 하는지
명확하지 않을 수 있다.

그래서 특정 interface를 지정하는 설정을 사용하는 경우가 있다.

---

# 62. Docker Network도 존재한다

Docker를 사용하면:

```text
docker0
```

같은 virtual network interface가 생길 수 있다.

컨테이너 내부와 host 사이에 별도의 network namespace가 존재할 수 있다.

이 내용은 Chapter 9에서 더 자세히 다룬다.

---

# 63. Robot Network Debugging 순서

Network 문제는 다음 순서로 좁히는 것이 좋다.

```text
1. Cable / Wi-Fi 연결?
       ↓
2. Interface UP?
       ↓
3. IP address 있음?
       ↓
4. Subnet 맞음?
       ↓
5. Route 있음?
       ↓
6. ping 됨?
       ↓
7. 필요한 port/protocol 열려 있음?
       ↓
8. Application 정상?
       ↓
9. ROS_DOMAIN_ID?
       ↓
10. DDS / Multicast / RMW?
```

---

# 64. Vision60 사례 1: Robot Wi-Fi 직접 연결

예:

```text
MacBook
192.168.0.50
      │
      │ Vision60 Wi-Fi
      ▼
Xavier
192.168.0.18
```

Mac에서:

```bash
ping 192.168.0.18
```

이 되고:

```bash
ssh user@192.168.0.18
```

가 된다면 직접 robot에 접속할 수 있다.

---

# 65. Vision60 사례 2: 회사 Wi-Fi에서도 접속하고 싶다

목표:

```text
MacBook
Company Wi-Fi
     │
     ▼
Company Network
     │
     ▼
Orin
     │
     ▼
Xavier
```

이 경우 필요한 것은 단순 Wi-Fi 연결이 아니라:

```text
Routing
Firewall
IP Forwarding
Network Policy
```

등이다.

---

# 66. Vision60 사례 3: Orin만 회사 Wi-Fi 연결

예:

```text
Orin
eth0 → Xavier
wlan0 → Company Wi-Fi
```

Orin에서는:

```text
Robot Network
+
Internet
```

을 동시에 사용할 수 있다.

하지만 Xavier가 자동으로 인터넷을 쓰게 되는 것은 아니다.

---

# 67. Xavier도 인터넷을 쓰게 하려면

Orin에서:

```text
IP forwarding
NAT
```

를 설정해야 할 수 있다.

구조:

```text
Xavier
   │
   ▼
Orin
   │
   ▼
Company Wi-Fi
   │
   ▼
Internet
```

---

# 68. 보안상 주의

Robot 내부 network를 회사망이나 인터넷에 연결하면
편리하지만 공격 surface도 커진다.

주의할 것:

```text
Default Password
Open SSH
Unused Ports
Weak Firewall
Old Packages
Exposed ROS 2 traffic
```

등이다.

---

# 69. SSH Key

Password 대신 SSH key authentication을 사용할 수 있다.

구조:

```text
Laptop
Private Key
   │
   ▼
Jetson
Public Key
```

일반적으로 password-only 방식보다 안전하고 편리하다.

---

# 70. SSH Key 생성

예:

```bash
ssh-keygen
```

그 후 public key를 remote machine에 등록한다.

예:

```bash
ssh-copy-id user@192.168.10.10
```

---

# 71. `.ssh/config`

여러 robot을 관리할 때:

```text
~/.ssh/config
```

에 설정을 넣으면 편하다.

예:

```text
Host vision60
    HostName 192.168.0.18
    User robot
```

그 후:

```bash
ssh vision60
```

처럼 접속 가능하다.

---

# 72. Latency 측정

`ping` 결과에서:

```text
time=2.1 ms
```

같은 값을 볼 수 있다.

이것은 round-trip time이다.

로봇에서는 latency가 일정한지도 중요하다.

---

# 73. Packet Loss

Wi-Fi 환경에서는 packet loss가 발생할 수 있다.

`ping`에서:

```text
10% packet loss
```

같은 결과를 볼 수 있다.

LiDAR/ROS 2 통신 품질에 영향을 줄 수 있다.

---

# 74. Jitter

Network latency가 계속 변하는 것을 jitter라고 볼 수 있다.

예:

```text
2 ms
3 ms
2 ms
40 ms
3 ms
```

로봇에서는 평균 latency뿐 아니라
jitter도 중요한 문제다.

---

# 75. Bandwidth 확인

Network bandwidth가 충분하지 않으면
camera나 LiDAR data가 밀릴 수 있다.

예:

```text
Camera = 300 Mbps
LiDAR = 100 Mbps
Other traffic = 200 Mbps
```

1 Gbps Ethernet이라도 실제 usable bandwidth와 overhead를 고려해야 한다.

---

# 76. `ethtool`

Ethernet link 상태를 확인할 때:

```bash
ethtool eth0
```

를 사용할 수 있다.

예:

```text
Speed: 1000Mb/s
Duplex: Full
Link detected: yes
```

---

# 77. Full Duplex

Full duplex는 송신/수신을 동시에 할 수 있다는 의미다.

```text
Jetson → LiDAR
Jetson ← LiDAR
```

를 동시에 처리할 수 있다.

---

# 78. `nmcli`

Ubuntu NetworkManager 환경에서는:

```bash
nmcli
```

를 자주 사용한다.

`nmcli`는:

**NetworkManager Command Line Interface**

이다.

예:

```bash
nmcli con show
```

여기서:

```text
con
```

은:

```text
connection
```

의 shorthand다.

---

# 79. Network Connection과 Interface는 다르다

`nmcli`에서는:

```text
Device
Connection Profile
```

을 구분한다.

예:

```text
Device:
eth0

Connection:
Wired connection 1
```

하나의 device에 여러 connection profile이 있을 수 있다.

---

# 80. Static IP 설정의 개념

예를 들어 `eth0`에:

```text
192.168.10.10/24
```

를 static으로 설정하면:

```text
Jetson eth0
      │
      ▼
Robot Network
```

에 항상 같은 주소로 참여할 수 있다.

실제 설정은 NetworkManager나 netplan 등을 사용할 수 있다.

---

# 81. NetworkManager

Ubuntu에서 network interface와 connection profile을 관리하는 service다.

명령:

```bash
nmcli
```

GUI가 있다면 graphical network settings도 NetworkManager를 사용할 수 있다.

---

# 82. Netplan

Ubuntu에서는 network configuration에:

```text
/etc/netplan/
```

을 사용하는 경우도 있다.

예:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      addresses:
        - 192.168.10.10/24
```

환경에 따라 NetworkManager나 systemd-networkd backend를 사용할 수 있다.

---

# 83. Network 변경 시 주의

SSH로 remote machine에 접속한 상태에서 network 설정을 바꾸면:

```text
SSH disconnect
```

될 수 있다.

특히 remote robot에서:

```text
IP
Route
Interface
```

를 바꿀 때는 physical access 가능 여부를 먼저 확인하는 것이 좋다.

---

# 84. Robot Network Architecture 문서화

실제 robot system에서는 IP를 문서화하는 것이 좋다.

예:

| Device | Interface | IP | Role |
|---|---|---|---|
| Xavier | eth0 | 192.168.10.10 | Robot compute |
| Orin | eth0 | 192.168.10.11 | AI compute |
| LiDAR | eth | 192.168.10.20 | Point cloud |
| Laptop | Wi-Fi | DHCP | Operator |
| Router | LAN | 192.168.10.1 | Gateway |

이런 표가 있으면 troubleshooting이 훨씬 쉬워진다.

---

# 85. ROS 2 Multi-Computer 예제

```text
Xavier
192.168.10.10
ROS_DOMAIN_ID=123
RMW=CycloneDDS

      │
      │ Ethernet
      ▼

Orin
192.168.10.11
ROS_DOMAIN_ID=123
RMW=CycloneDDS
```

Xavier:

```text
LiDAR Driver
FAST-LIO2
```

Orin:

```text
Vision AI
Navigation
```

처럼 역할을 나눌 수 있다.

---

# 86. Topic은 Network를 넘어갈 수 있다

예:

```text
Xavier
/odometry Publisher
      │
      │ ROS 2 / DDS
      ▼
Orin
/odometry Subscriber
```

application code에서는 같은 topic처럼 보이지만
실제로는 Ethernet network를 통해 packet이 전달될 수 있다.

---

# 87. ROS 2가 안 보일 때 확인

두 computer에서:

```bash
echo $ROS_DOMAIN_ID
```

확인.

```bash
echo $RMW_IMPLEMENTATION
```

확인.

```bash
ip addr
```

확인.

```bash
ping <other-machine>
```

확인.

그리고:

```bash
ros2 node list
```

로 discovery를 본다.

---

# 88. Mini Practice 1

현재 Jetson에서:

```bash
ip addr
```

실행.

다음 질문에 답한다.

```text
eth0 IP는?
wlan0 IP는?
lo는 무엇인가?
```

---

# 89. Mini Practice 2

```bash
ip route
```

실행.

찾아볼 것:

```text
default route
robot subnet route
어느 interface가 사용되는가?
```

---

# 90. Mini Practice 3

다른 device에:

```bash
ping <IP>
```

실행.

확인:

```text
latency
packet loss
```

---

# 91. Mini Practice 4

Ethernet device가 있다면:

```bash
ethtool eth0
```

확인.

찾아볼 것:

```text
Speed
Duplex
Link detected
```

---

# 92. Mini Practice 5

NetworkManager 환경:

```bash
nmcli device
```

```bash
nmcli con show
```

를 실행한다.

비교:

```text
Device
vs
Connection
```

---

# 93. Mini Practice 6

ROS 2 두 machine에서:

```bash
echo $ROS_DOMAIN_ID
```

```bash
echo $RMW_IMPLEMENTATION
```

확인하고,

```bash
ros2 node list
```

결과를 비교한다.

---

# 94. 오늘의 핵심

Robot network를 볼 때 다음 네 층으로 생각한다.

```text
Application
ROS 2 / SSH

      ↓

Transport
TCP / UDP

      ↓

Network
IP / Routing

      ↓

Link
Ethernet / Wi-Fi
```

그리고 실제 장치는:

```text
IP
Subnet
Gateway
Route
Interface
```

를 가지고 있다.

---

# 95. 반드시 구분할 것

```text
Ethernet ≠ Internet

IP Address ≠ MAC Address

Switch ≠ Router

Routing ≠ NAT

DHCP ≠ DNS

Wi-Fi ≠ Internet

ping OK ≠ ROS 2 OK

SSH ≠ Remote Desktop

ROS_DOMAIN_ID ≠ IP Address
```

---

# 96. Vision60 Network Mental Model

최종적으로 다음처럼 생각할 수 있다.

```text
                    Company Network
                           │
                         Wi-Fi
                           │
                           ▼
                    ┌────────────┐
                    │    Orin    │
                    │            │
                    │ wlan0      │
                    │ eth0       │
                    └─────┬──────┘
                          │
                       Ethernet
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
       Xavier           LiDAR          Other Device
   192.168.10.10    192.168.10.20

          │
          ▼
       ROS 2
          │
          ▼
      Robot Control
```

이 구조에서 문제가 생겼을 때:

```text
Hardware?
Interface?
IP?
Subnet?
Route?
Firewall?
DDS?
ROS 2?
```

순서로 확인하면 된다.

---

# Next Chapter

## Chapter 9. Docker on Jetson

다음 Chapter에서는 Jetson에서 Docker를 왜 사용하는지 살펴본다.

- Container와 Virtual Machine은 무엇이 다른가?
- Docker image와 container는 무엇이 다른가?
- `docker run`은 무엇을 하는가?
- Volume과 bind mount는 무엇인가?
- Container 안에서 USB/CAN/GPU를 어떻게 접근하는가?
- `--network host`는 무엇인가?
- Docker 안에서 ROS 2가 왜 안 보일 수 있는가?
- `linux/amd64`와 `linux/arm64`는 왜 중요한가?
- JetPack/CUDA와 container compatibility는 어떻게 맞추는가?
- NVIDIA Container Runtime은 무엇인가?
- Jetson에서 ROS 2 + CUDA + TensorRT container를 어떻게 구성하는가?

Chapter 9에서는 지금까지 배운:

```text
ARM64
+
Linux
+
Jetson
+
Network
+
ROS 2
```

를 하나의 reproducible software environment로 묶는 방법을 다룬다.