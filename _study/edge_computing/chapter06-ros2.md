---
title: "Chapter 6. ROS 2 as a Robotics Middleware"
importance: 7
---

> **Goal:** ROS 2가 로봇 software architecture에서 정확히 어느 위치에 있는지 이해한다.
> Node, Topic, Service, Action뿐 아니라 `rclcpp → RMW → DDS → UDP/IP → Ethernet`으로 이어지는
> 실제 communication stack을 이해하는 것이 목표다.

---

# 1. ROS 2는 정확히 무엇인가?

ROS는:

**Robot Operating System**

의 약자다.

하지만 이름과 달리 Ubuntu나 Windows 같은 Operating System은 아니다.

ROS 2는 로봇 software를 개발하기 위한:

> **Middleware + Framework + Tools + Libraries**

의 집합이라고 보는 것이 가장 정확하다.

예를 들어 ROS 2는 다음 기능을 제공한다.

```text
ROS 2

├── Node
├── Topic
├── Service
├── Action
├── Parameter
├── Logging
├── Launch
├── Discovery
├── Message Definition
├── CLI Tools
└── Communication Middleware
```

---

# 2. ROS 2는 왜 필요한가?

로봇에는 여러 프로그램이 동시에 실행된다.

예를 들어 Vision60을 단순화하면:

```text
LiDAR Driver
IMU Driver
FAST-LIO2
Navigation
Controller
Camera
Visualization
```

ROS 2가 없다면 각각의 프로그램 사이 통신 방법을 직접 만들어야 한다.

```text
LiDAR Program
      │
      │ 직접 socket 구현?
      ▼
SLAM Program
      │
      │ 직접 protocol 구현?
      ▼
Navigation
```

ROS 2를 사용하면:

```text
LiDAR Node
     │
     │ /points
     ▼
FAST-LIO2
     │
     │ /odometry
     ▼
Navigation
```

처럼 공통 communication system을 사용할 수 있다.

---

# 3. ROS 2는 Protocol인가?

단순히:

```text
ROS 2 = Protocol
```

이라고 하는 것은 정확하지 않다.

ROS 2는 TCP나 UDP 같은 하나의 network protocol보다 훨씬 큰 system이다.

비교하면:

```text
ROS 2
→ Robotics middleware/framework

DDS
→ Data-centric middleware technology

UDP/TCP
→ Transport protocol

IP
→ Network protocol

Ethernet
→ Link-layer networking technology
```

따라서 이들은 같은 level의 개념이 아니다.

---

# 4. 전체 Communication Stack

ROS 2 node가 다른 computer의 node로 message를 보낸다고 생각해보자.

전체 구조를 단순화하면:

```text
Application
     │
     ▼
ROS 2 Node
     │
     ▼
rclcpp / rclpy
     │
     ▼
rcl
     │
     ▼
RMW
     │
     ▼
DDS Implementation
     │
     ▼
UDP / TCP 등
     │
     ▼
IP
     │
     ▼
Ethernet / Wi-Fi
     │
     ▼
Physical Network
```

이 그림이 Chapter 6에서 가장 중요하다.

---

# 5. Node란?

Node는 ROS 2에서 하나의 기능을 담당하는 실행 단위다.

예:

```text
LiDAR Driver Node
IMU Driver Node
FAST-LIO2 Node
Navigation Node
Controller Node
```

각 node가 특정 역할을 담당하도록 software를 분리할 수 있다.

예:

```text
             Robot

        LiDAR Node
             │
             ▼
        FAST-LIO2
             │
             ▼
       Navigation
             │
             ▼
        Controller
```

---

# 6. Process와 Node는 같은가?

항상 같지는 않다.

Chapter 3에서:

> Process = 실행 중인 program의 instance

라고 배웠다.

ROS 2 Node는 ROS graph의 논리적 참여 단위다.

간단한 경우:

```text
Process
   │
   └── Node A
```

처럼 하나의 process에 하나의 node가 있을 수 있다.

하지만 composition을 사용하면:

```text
One Process

├── Node A
├── Node B
└── Node C
```

처럼 여러 node를 하나의 process 안에서 실행할 수도 있다.

따라서:

```text
Node ≠ Process
```

이다.

---

# 7. ROS Graph

실행 중인 ROS 2 system의 node와 communication 관계를:

```text
ROS Graph
```

라고 한다.

예:

```text
/lidar_driver
      │
      │ /points
      ▼
/fastlio
      │
      │ /odometry
      ▼
/navigation
```

ROS 2 command로 node 확인:

```bash
ros2 node list
```

---

# 8. Topic

Topic은 ROS 2에서 지속적인 data stream을 전달할 때 많이 사용한다.

예:

```text
LiDAR Point Cloud
IMU
Camera Image
Odometry
Joint State
```

구조:

```text
Publisher
    │
    │ Topic
    ▼
Subscriber
```

---

# 9. Publisher와 Subscriber

데이터를 보내는 쪽:

```text
Publisher
```

받는 쪽:

```text
Subscriber
```

예:

```text
LiDAR Driver
 Publisher
     │
     │ /points_raw
     ▼
FAST-LIO2
 Subscriber
```

---

# 10. Topic은 1:1 통신만 가능한가?

아니다.

하나의 topic을 여러 subscriber가 받을 수 있다.

```text
             /points

LiDAR ─────────┬──── FAST-LIO2
Publisher      │
               ├──── RViz
               │
               └──── Recorder
```

또한 ROS 2의 pub/sub 모델은 단순한 1:1 socket 구조와 다르다.

---

# 11. Topic 확인 명령어

현재 topic 목록:

```bash
ros2 topic list
```

특정 topic의 message 확인:

```bash
ros2 topic echo /imu
```

Topic 정보:

```bash
ros2 topic info /imu
```

Message frequency:

```bash
ros2 topic hz /imu
```

예:

```text
average rate: 200 Hz
```

라면 대략 초당 200개의 message가 들어온다는 뜻이다.

---

# 12. Message Type

Topic에는 아무 data나 보낼 수 있는 것이 아니다.

Message type이 정의되어 있다.

예:

```text
sensor_msgs/msg/Imu
sensor_msgs/msg/PointCloud2
nav_msgs/msg/Odometry
geometry_msgs/msg/Twist
```

예:

```text
/imu
   │
   └── sensor_msgs/msg/Imu
```

Publisher와 subscriber는 compatible한 message type을 사용해야 한다.

---

# 13. Service

Service는 request-response 방식이다.

```text
Client
   │
   │ Request
   ▼
Server
   │
   │ Response
   ▼
Client
```

예:

```text
"Map을 저장해줘"
       │
       ▼
Map Server
       │
       ▼
"저장 완료"
```

Topic과 달리 지속적인 stream보다는 특정 요청과 응답에 적합하다.

---

# 14. Topic vs Service

```text
Topic

Publisher
   │
   │ data
   │ data
   │ data
   ▼
Subscriber
```

반면:

```text
Service

Client
   │ request
   ▼
Server
   │ response
   ▼
Client
```

| Topic | Service |
|---|---|
| 지속적 data stream | 요청/응답 |
| Publisher/Subscriber | Client/Server |
| Sensor data에 적합 | 명령/조회에 적합 |
| 비동기적인 pub/sub | request-response |

---

# 15. Action

Action은 시간이 오래 걸리는 작업을 요청할 때 사용한다.

예:

```text
"목표 위치까지 이동해"
```

Navigation은 즉시 끝나지 않는다.

그래서:

```text
Goal
Feedback
Result
```

가 필요하다.

구조:

```text
Action Client
     │
     │ Goal
     ▼
Action Server
     │
     ├── Feedback
     ├── Feedback
     ├── Feedback
     │
     ▼
   Result
```

---

# 16. Service와 Action 차이

예를 들어:

```text
Service
"현재 battery level 알려줘"
→ 빠른 응답
```

반면:

```text
Action
"저 위치까지 이동해"
→ 수 초~수 분 소요
→ 중간 progress 필요
→ 취소 가능
```

Navigation에서 Action을 많이 사용하는 이유다.

---

# 17. Topic / Service / Action

| 종류 | 용도 | 예 |
|---|---|---|
| Topic | 지속적인 data | LiDAR, IMU, Odometry |
| Service | 빠른 요청/응답 | 설정 변경, 상태 요청 |
| Action | 장시간 작업 | Navigation goal |

---

# 18. rclcpp란?

ROS 2 C++ code에서 자주 보는:

```cpp
#include <rclcpp/rclcpp.hpp>
```

여기서 `rclcpp`는:

> ROS Client Library for C++

이다.

C++ 개발자가 ROS 2 기능을 사용할 수 있도록 API를 제공한다.

예:

```cpp
rclcpp::Node
rclcpp::Publisher
rclcpp::Subscription
```

---

# 19. rclpy란?

Python에서는:

```python
import rclpy
```

를 사용한다.

`rclpy`는:

> ROS Client Library for Python

이다.

구조:

```text
C++ Application
      │
    rclcpp
      │
      ▼
     ROS 2
```

```text
Python Application
      │
    rclpy
      │
      ▼
     ROS 2
```

---

# 20. rcl이란?

`rclcpp`와 `rclpy` 아래에는 공통적인 C 기반 ROS Client Library layer인:

```text
rcl
```

이 존재한다.

단순화하면:

```text
C++            Python
 │               │
rclcpp          rclpy
  \              /
   \            /
        rcl
         │
         ▼
        RMW
```

이렇게 여러 language client library가 공통 lower layer를 사용할 수 있다.

---

# 21. RMW란?

RMW는:

**ROS Middleware Interface**

이다.

ROS 2와 실제 middleware implementation 사이의 abstraction layer다.

```text
ROS 2
  │
  ▼
RMW
  │
  ▼
DDS Implementation
```

RMW 덕분에 ROS application code를 크게 바꾸지 않고
다른 middleware implementation을 선택할 수 있다.

---

# 22. DDS란?

DDS는:

**Data Distribution Service**

이다.

분산 system에서 data를 publish/subscribe 방식으로 전달하기 위한 middleware standard다.

ROS 2는 DDS 계열 middleware를 널리 사용한다.

예:

```text
Node A
Publisher
   │
   ▼
DDS
   │
   ▼
DDS
   │
   ▼
Node B
Subscriber
```

DDS는 단순히 "network packet을 보내는 library" 이상의 역할을 한다.

예:

```text
Discovery
Publish / Subscribe
QoS
Serialization
Data Distribution
```

등을 담당한다.

---

# 23. DDS Implementation

DDS는 specification이고 실제 software implementation은 여러 개가 있다.

ROS 2에서 대표적으로 볼 수 있는 것:

```text
Cyclone DDS
Fast DDS
```

등이다.

예:

```text
ROS 2
   │
   ▼
RMW
   │
   ├── rmw_cyclonedds_cpp
   │        │
   │        ▼
   │    Cyclone DDS
   │
   └── rmw_fastrtps_cpp
            │
            ▼
         Fast DDS
```

---

# 24. RMW Implementation 선택

환경 변수에서 이런 것을 볼 수 있다.

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

의미:

```text
ROS 2가 사용할 RMW implementation
          ↓
rmw_cyclonedds_cpp
          ↓
Cyclone DDS
```

이다.

Vision60 같은 multi-computer robot에서 DDS 설정을 맞추는 것이 중요할 수 있다.

---

# 25. DDS와 TCP/UDP는 같은 것인가?

아니다.

DDS는 더 위 layer의 middleware다.

```text
ROS 2
  │
DDS
  │
Transport
  │
UDP / TCP / Shared Memory 등
  │
IP
  │
Ethernet
```

구체적인 DDS implementation과 설정에 따라 transport 방식은 달라질 수 있다.

따라서:

```text
DDS ≠ UDP
DDS ≠ TCP
```

이다.

---

# 26. ROS 2와 TCP/IP는 동급인가?

아니다.

예를 들어 network communication이라면:

```text
ROS 2 Application
       ↓
DDS Middleware
       ↓
UDP/IP
       ↓
Ethernet
```

처럼 여러 layer가 겹쳐 있다.

따라서 software architecture diagram에서:

```text
ROS 2
TCP
HTTP
```

를 모두 같은 종류의 protocol로 나열하면 의미가 모호해질 수 있다.

더 정확하게 분류하면:

```text
ROS 2
→ Robotics middleware/framework

DDS
→ Middleware

HTTP
→ Application protocol

TCP / UDP
→ Transport protocol

IP
→ Network protocol

Ethernet
→ Link-layer technology
```

이다.

---

# 27. 인터넷이 없어도 ROS 2가 되는가?

**된다.**

인터넷과 local network는 다른 개념이다.

예:

```text
Xavier
192.168.10.10
     │
     │ Ethernet
     ▼
Orin
192.168.10.11
```

두 컴퓨터가 local network에서 서로 통신할 수 있다면
인터넷이 없어도 ROS 2 communication이 가능하다.

```text
Internet
   X

Xavier ←──── Ethernet ────→ Orin
             ROS 2
```

---

# 28. Network 자체가 없어도 ROS 2가 되는가?

같은 computer 안의 ROS 2 node끼리도 통신할 수 있다.

```text
            Jetson

┌──────────────────────────┐
│                          │
│ LiDAR Node               │
│      │                   │
│      │ ROS 2             │
│      ▼                   │
│ FAST-LIO2 Node           │
│                          │
└──────────────────────────┘
```

외부 Ethernet이나 Wi-Fi 연결이 없어도 된다.

즉:

```text
Internet 없어도 가능
External network 없어도 같은 machine에서는 가능
```

이다.

---

# 29. 그렇다면 같은 컴퓨터에서도 network stack을 쓰나?

경우에 따라 middleware가:

```text
Loopback
Shared Memory
Inter-process communication
```

등을 사용할 수 있다.

구체적인 통신 경로는 RMW/DDS implementation과 설정,
그리고 node들이 같은 process인지 다른 process인지에 따라 달라질 수 있다.

따라서:

> ROS 2 communication = 무조건 Ethernet packet

이라고 생각하면 안 된다.

---

# 30. Intra-process Communication

두 node가 같은 process 안에 있다면:

```text
Process

Node A
  │
  ▼
Node B
```

ROS 2의 intra-process communication 기능을 사용하여
불필요한 serialization/copy를 줄이는 최적화를 사용할 수 있다.

이는 camera나 point cloud처럼 큰 data에서 중요할 수 있다.

---

# 31. Serialization

ROS message는 network로 보내기 위해 전송 가능한 byte representation으로 변환되어야 한다.

이를:

```text
Serialization
```

이라고 한다.

예:

```text
ROS Message

position.x = 1.2
position.y = 3.4

       │
       ▼
Serialization
       │
       ▼
Bytes
       │
       ▼
Network
```

수신 측에서는 반대로 deserialize한다.

---

# 32. Discovery

ROS 2에서는 node들이 서로를 자동으로 찾는 **Discovery** 기능이 있다.

예:

```text
Xavier
/imu_driver

      ↕ discovery

Orin
/fastlio
```

사용자가 매번:

```text
FAST-LIO2의 IP는 192.168.10.20이다
```

라고 직접 연결 정보를 코드에 넣지 않아도
middleware가 participant와 endpoint를 찾을 수 있다.

---

# 33. ROS_DOMAIN_ID

같은 physical network에 여러 ROS 2 system이 있을 수 있다.

예:

```text
Robot A
Robot B
Robot C
```

모든 node가 서로 discovery되면 문제가 생길 수 있다.

그래서 ROS 2에는:

```text
ROS_DOMAIN_ID
```

가 있다.

예:

```bash
export ROS_DOMAIN_ID=123
```

---

# 34. Domain 분리

예:

```text
Robot A
ROS_DOMAIN_ID=10

Robot B
ROS_DOMAIN_ID=20
```

이면 서로 다른 DDS domain으로 분리된다.

반대로 Xavier와 Orin이 같은 robot에서 통신해야 한다면:

```text
Xavier
ROS_DOMAIN_ID=123

Orin
ROS_DOMAIN_ID=123
```

처럼 맞추는 것이 일반적이다.

---

# 35. QoS란?

QoS는:

**Quality of Service**

이다.

ROS 2/DDS에서는 message 전달 특성을 설정할 수 있다.

대표적으로:

```text
Reliability
Durability
History
Depth
Deadline
Liveliness
```

등이 있다.

---

# 36. Reliability

대표적으로:

```text
Reliable
Best Effort
```

가 있다.

## Reliable

가능한 한 message 전달을 보장하려고 한다.

```text
Message Lost
    ↓
Recovery / Retransmission behavior
```

---

## Best Effort

일부 message가 유실되어도 최신 data를 빠르게 받는 것을 우선할 수 있다.

```text
1 2 3 4 5 6

Network loss

1 2   4 5 6
```

sensor stream에서는 모든 과거 message보다 최신 data가 더 중요한 경우가 있다.

---

# 37. SensorDataQoS

LiDAR, IMU, camera 같은 sensor에서는:

```text
SensorDataQoS
```

profile을 자주 사용한다.

C++ 예:

```cpp
rclcpp::SensorDataQoS()
```

Sensor data는:

```text
High Frequency
Continuous
Latest Data Important
```

라는 특징이 있기 때문에 low-latency와 best-effort 성격의 QoS가 적합한 경우가 많다.

---

# 38. QoS가 안 맞으면?

Publisher와 Subscriber의 QoS가 compatible하지 않으면
topic 이름이 맞아도 communication이 되지 않을 수 있다.

예:

```text
LiDAR Publisher
      │
      │ QoS mismatch
      X
      │
FAST-LIO2 Subscriber
```

그래서:

> "topic은 보이는데 data가 안 들어온다"

면 QoS도 확인해야 한다.

---

# 39. History와 Depth

QoS에서는 message를 얼마나 보관할지도 정할 수 있다.

예:

```text
Keep Last
Depth = 10
```

이라면 최근 message를 일정 개수 queue에 유지하는 식이다.

Sensor가 subscriber 처리 속도보다 빠르면 queue 크기도 영향을 줄 수 있다.

---

# 40. LiDAR → FAST-LIO2 전체 흐름

이제 실제 SLAM pipeline을 layer별로 보자.

```text
LiDAR Hardware
      │
      │ Ethernet / UDP
      ▼
LiDAR Driver
      │
      │ PointCloud2
      ▼
ROS 2 Publisher
      │
      ▼
rclcpp
      │
      ▼
RMW
      │
      ▼
DDS
      │
      ▼
RMW
      │
      ▼
rclcpp
      │
      ▼
FAST-LIO2 Subscriber
      │
      ▼
Point Cloud Processing
```

이제 "LiDAR가 ROS 2로 데이터를 보낸다"라는 말을 더 정확히 이해할 수 있다.

실제로 LiDAR hardware가 ROS 2를 직접 사용하는 것이 아니라:

```text
LiDAR
   ↓
Vendor Protocol / UDP
   ↓
Driver
   ↓
ROS 2 Message
```

로 변환되는 것이다.

---

# 41. IMU Callback

FAST-LIO2 코드에서:

```cpp
imu_cbk(...)
```

같은 이름을 볼 수 있다.

`cbk`는 보통:

```text
callback
```

을 줄여 쓴 이름이다.

구조:

```text
IMU Publisher
      │
      │ /imu
      ▼
FAST-LIO2 Subscription
      │
      ▼
imu_cbk()
```

새로운 IMU message가 도착하면 callback 함수가 실행되는 구조다.

---

# 42. Callback이란?

Callback은 특정 event가 발생했을 때 호출되는 함수다.

예:

```cpp
void imu_callback(const Imu::SharedPtr msg)
{
    // process IMU
}
```

개념:

```text
Waiting...

IMU Message Arrives
       │
       ▼
Callback 실행
```

ROS 2 subscriber programming의 핵심 개념이다.

---

# 43. Executor

그렇다면 callback을 누가 실행할까?

ROS 2에서는:

```text
Executor
```

가 callback 실행을 관리한다.

단순화하면:

```text
ROS Events
   │
   ├── IMU Message
   ├── LiDAR Message
   ├── Timer
   └── Service Request
          │
          ▼
       Executor
          │
          ▼
       Callback
```

---

# 44. SingleThreadedExecutor

한 thread에서 callback들을 처리한다.

```text
Thread

IMU callback
    ↓
LiDAR callback
    ↓
Timer callback
```

한 callback이 오래 걸리면 다른 callback 실행이 지연될 수 있다.

---

# 45. MultiThreadedExecutor

여러 thread를 사용해 callback을 처리할 수 있다.

```text
Thread 1 → IMU callback

Thread 2 → LiDAR callback

Thread 3 → Timer callback
```

하지만 동시에 shared data에 접근한다면 synchronization 문제가 생길 수 있으므로 주의해야 한다.

---

# 46. ROS 2 Launch

여러 node를 매번 하나씩 실행하기 어렵다.

예:

```bash
ros2 run lidar driver
ros2 run imu driver
ros2 run fastlio mapping
ros2 run nav2 ...
```

Launch system을 사용하면 여러 node와 parameter를 함께 실행할 수 있다.

```bash
ros2 launch vision60_bringup bringup.launch.py
```

구조:

```text
Launch
  │
  ├── LiDAR Node
  ├── FAST-LIO2 Node
  ├── TF Node
  └── Navigation Node
```

---

# 47. Parameter

ROS 2 node는 parameter를 가질 수 있다.

예:

```text
lidar_topic
imu_topic
frame_id
mapping_rate
```

확인:

```bash
ros2 param list
```

특정 값:

```bash
ros2 param get /node_name parameter_name
```

Parameter file을 YAML로 관리하는 경우도 많다.

---

# 48. TF란?

로봇에는 여러 coordinate frame이 있다.

예:

```text
map
odom
base_link
lidar_link
imu_link
```

TF는 이 frame 사이의 transform을 관리한다.

예:

```text
map
 │
 ▼
odom
 │
 ▼
base_link
 │
 ├── lidar_link
 └── imu_link
```

SLAM과 Navigation에서는 매우 중요한 ROS subsystem이다.

---

# 49. Topic과 TF의 차이

TF도 내부적으로 ROS communication을 사용하지만,
개념적으로는:

```text
Topic
→ 일반적인 data stream

TF
→ Coordinate frame relationship
```

을 관리하기 위한 subsystem이다.

예:

```text
/imu
→ sensor measurement

odom → base_link
→ coordinate transform
```

---

# 50. ROS 2 CLI는 무엇을 하는가?

우리가 사용하는:

```bash
ros2 node list
ros2 topic list
ros2 topic echo
ros2 service list
ros2 action list
```

등은 ROS 2 system을 확인하고 조작하기 위한 CLI tool이다.

즉 `ros2`라는 명령어 자체가 ROS 2 전체는 아니다.

```text
ROS 2 System
     │
     └── ros2 CLI
```

이다.

---

# 51. ROS 2 Package

관련된 code, config, launch file 등을 package 단위로 관리한다.

예:

```text
my_robot_package/

├── src/
├── include/
├── launch/
├── config/
├── package.xml
└── CMakeLists.txt
```

Package는 ROS 2 software를 구성하는 기본적인 배포/빌드 단위다.

---

# 52. Workspace

여러 package를 하나의 workspace에서 관리할 수 있다.

예:

```text
ros2_ws/

├── src/
│   ├── fast_lio/
│   ├── vision60_bringup/
│   └── odom_tf_adapter/
│
├── build/
├── install/
└── log/
```

보통 source code는:

```text
src/
```

아래에 있다.

---

# 53. `colcon build`

ROS 2 workspace를 build할 때:

```bash
colcon build
```

를 사용한다.

결과:

```text
src
 │
 │ colcon build
 ▼
build
install
log
```

이 생성된다.

---

# 54. 왜 `install/setup.bash`를 source할까?

Build 후:

```bash
source install/setup.bash
```

를 실행한다.

이유는 현재 shell이 새로 build한 package의 위치를 알 수 있도록
environment variable을 설정하기 위해서다.

Chapter 3의 `source`와 연결된다.

```text
colcon build
      │
      ▼
install/
      │
      ▼
source install/setup.bash
      │
      ▼
Current Shell
ROS package 발견 가능
```

---

# 55. Underlay와 Overlay

예를 들어:

```bash
source /opt/ros/humble/setup.bash
```

먼저 system ROS 2를 source한다.

그 다음:

```bash
source ~/vision60_ws/install/setup.bash
```

를 source한다.

구조:

```text
Ubuntu
  │
  ▼
ROS 2 Humble
/opt/ros/humble
  │
  ▼
Vision60 Workspace
~/vision60_ws/install
```

위 workspace가 아래 environment 위에 overlay되는 형태로 볼 수 있다.

---

# 56. Vision60에서 전체 구조

지금까지 배운 내용을 Vision60에 연결하면:

```text
                       Vision60

 LiDAR ── Ethernet ──► LiDAR Driver Node
                              │
                              │ PointCloud2
                              ▼
 IMU ─────────────────► IMU Driver Node
                              │
                              │ Imu
                              ▼
                       ┌─────────────┐
                       │ FAST-LIO2   │
                       └──────┬──────┘
                              │
                         Odometry
                              │
                              ▼
                         Navigation
                              │
                              ▼
                         Controller
```

그리고 communication 내부를 확대하면:

```text
ROS Node
   │
rclcpp
   │
rcl
   │
RMW
   │
Cyclone DDS
   │
UDP / Shared Memory / etc.
   │
Linux
   │
Ethernet / Local Machine
```

---

# 57. ROS 2 Debugging 순서

FAST-LIO2에 sensor data가 안 들어온다고 하자.

무작정 FAST-LIO2 code부터 수정하지 않는다.

```text
1. Hardware connected?
       ↓
2. Linux sees device/network?
       ↓
3. Driver running?
       ↓
4. ROS node exists?
       ↓
5. Topic exists?
       ↓
6. Topic data arriving?
       ↓
7. Message type correct?
       ↓
8. QoS compatible?
       ↓
9. ROS_DOMAIN_ID correct?
       ↓
10. RMW/DDS/network correct?
       ↓
11. FAST-LIO2 callback running?
```

이 순서로 좁혀갈 수 있다.

---

# 58. 실무 명령어

## Node

```bash
ros2 node list
```

```bash
ros2 node info /node_name
```

---

## Topic

```bash
ros2 topic list
```

```bash
ros2 topic echo /topic
```

```bash
ros2 topic info /topic
```

```bash
ros2 topic hz /topic
```

---

## Service

```bash
ros2 service list
```

---

## Action

```bash
ros2 action list
```

---

## Parameter

```bash
ros2 param list
```

---

## Interface

```bash
ros2 interface show sensor_msgs/msg/Imu
```

---

## Environment

```bash
echo $ROS_DOMAIN_ID
```

```bash
echo $RMW_IMPLEMENTATION
```

---

# 59. Mini Practice

Vision60/Jetson ROS 2 환경에서 다음을 실행해본다.

```bash
source /opt/ros/humble/setup.bash
```

workspace가 있다면:

```bash
source ~/vision60_ws/install/setup.bash
```

현재 node:

```bash
ros2 node list
```

Topic:

```bash
ros2 topic list
```

IMU topic을 찾아:

```bash
ros2 topic hz <imu-topic>
```

LiDAR topic을 찾아:

```bash
ros2 topic hz <lidar-topic>
```

그리고:

```bash
ros2 topic info <lidar-topic> --verbose
```

를 실행해 Publisher/Subscriber와 QoS 정보를 확인해본다.

마지막으로:

```bash
echo $ROS_DOMAIN_ID
```

```bash
echo $RMW_IMPLEMENTATION
```

을 확인한다.

---

# 60. 오늘의 핵심

가장 중요한 것은 ROS 2를 하나의 protocol로 생각하지 않는 것이다.

```text
┌─────────────────────────────┐
│ Robot Application           │
│ FAST-LIO2 / Navigation      │
├─────────────────────────────┤
│ ROS 2                       │
│ Node / Topic / Service      │
├─────────────────────────────┤
│ rclcpp / rclpy              │
├─────────────────────────────┤
│ rcl                         │
├─────────────────────────────┤
│ RMW                         │
├─────────────────────────────┤
│ DDS                         │
│ CycloneDDS / Fast DDS       │
├─────────────────────────────┤
│ UDP / TCP / Shared Memory   │
├─────────────────────────────┤
│ IP                          │
├─────────────────────────────┤
│ Ethernet / Wi-Fi            │
└─────────────────────────────┘
```

각 layer는 역할이 다르다.

---

# 61. 반드시 구분할 것

```text
ROS 2 ≠ Operating System

ROS 2 ≠ TCP

ROS 2 ≠ DDS

DDS ≠ UDP

Node ≠ Process

Topic ≠ Network Port

ROS_DOMAIN_ID ≠ IP Address

rclcpp ≠ ROS 2 전체

RMW ≠ DDS implementation 자체

Ethernet ≠ Internet
```

---

# 62. 한 문장으로 설명하기

누군가:

> "ROS 2가 뭐예요?"

라고 물으면 다음처럼 설명할 수 있다.

> **ROS 2 is a robotics middleware and framework that provides communication, tooling, and software abstractions for building distributed robot systems.**

조금 더 쉽게:

> **ROS 2 lets different robot programs communicate and work together without each program having to implement its own communication system.**

---

# 63. 지금까지 Chapter 연결

이제 Chapter 1~6이 하나의 stack으로 연결된다.

```text
Chapter 6
ROS 2
        ↑
Chapter 5
Ethernet / CAN / USB / PCIe
        ↑
Chapter 4
Jetson / JetPack
        ↑
Chapter 3
Linux
        ↑
Chapter 2
ARM64 / x86_64
        ↑
Chapter 1
CPU / GPU / RAM / Storage
```

즉 Vision60에서:

```text
FAST-LIO2가 ROS 2 Node로 실행된다
```

라는 한 문장 안에도 사실:

```text
C++ Application
      ↓
ROS 2
      ↓
Linux Process
      ↓
ARM64 Machine Code
      ↓
Jetson CPU
      ↓
RAM
```

라는 여러 layer가 숨어 있다.

---

# Next Chapter

## Chapter 7. CUDA & TensorRT for Robotics

다음 Chapter에서는 마지막으로 **Jetson의 GPU를 실제 software가 어떻게 사용하는지**를 다룬다.

- CPU code와 GPU code는 어떻게 다른가?
- CUDA kernel이란?
- Thread, Block, Grid란?
- 왜 GPU에는 수천 개의 thread를 만드는가?
- CPU RAM과 GPU memory는 어떻게 연결되는가?
- Jetson의 shared memory architecture에서는 무엇이 달라지는가?
- CUDA memory copy란?
- PyTorch의 `.to("cuda")`는 무엇을 하는가?
- FP32, FP16, INT8은 무엇인가?
- TensorRT가 model을 어떻게 최적화하는가?
- Latency와 Throughput은 무엇이 다른가?
- SLAM을 GPU로 옮기면 무조건 빨라지는가?
- Vision60에서 GPU를 어디에 쓰는 것이 효과적인가?

Chapter 7에서는 최종적으로:

```text
Sensor
   ↓
Jetson
   ├── CPU → ROS 2 / SLAM / Control
   │
   └── GPU → Vision / AI / Parallel Compute
              ↓
           TensorRT
```

까지 연결한다.