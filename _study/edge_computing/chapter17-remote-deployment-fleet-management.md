---
title: "Chapter 17. Remote Deployment & Fleet Management"
importance: 18
---

> **Goal:** 여러 대의 robot edge device를 원격으로 관리하는 기본 구조를 이해한다.
>
> Fleet, Device Identity, Provisioning, OTA Update, Container Registry,
> Versioning, Staged Rollout, Rollback, Remote Command, Telemetry,
> CI/CD, Device Group의 개념을 이해하고,
> Jetson/robot fleet을 실제 production 환경에서 어떻게 운영하는지 연결한다.

---

# 1. Robot 한 대와 Robot 100대는 완전히 다른 문제다

Robot이 한 대라면:

```text
Laptop
   │
   │ SSH
   ▼
Robot
```

직접 접속해서:

```bash
git pull
docker pull
systemctl restart ...
```

하면 될 수 있다.

하지만 robot이:

```text
10대
100대
1000대
```

가 되면 이런 방식은 유지하기 어렵다.

---

# 2. Fleet이란?

Fleet은:

> 중앙에서 함께 관리하는 device들의 집합

이다.

예:

```text
Fleet

├── Robot 001
├── Robot 002
├── Robot 003
├── Robot 004
└── ...
```

각 robot은 독립된 device지만
software version, configuration, health를 중앙에서 관리할 수 있다.

---

# 3. Fleet Management가 필요한 이유

여러 robot을 운영하면 다음 문제가 생긴다.

```text
어떤 robot이 어느 software version인가?
어떤 robot이 online인가?
어떤 robot이 update에 실패했는가?
어떤 robot이 disk가 거의 찼는가?
어떤 robot의 sensor가 죽었는가?
```

Fleet management는 이런 문제를 중앙에서 관리하기 위한 구조다.

---

# 4. 가장 단순한 Fleet Architecture

```text
                Developer / Operator

                        │
                        ▼
                Fleet Management
                     Server
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Robot A          Robot B          Robot C
```

중앙 system이 각 robot의 상태를 보고
필요한 command나 update를 전달한다.

---

# 5. Device Identity

Fleet에서는 각 robot이 고유한 identity를 가져야 한다.

예:

```text
vision60-001
vision60-002
vision60-003
```

각 robot을 구분할 수 있어야 한다.

---

# 6. Device Identity와 Hostname은 다르다

Hostname:

```text
vision60-001
```

도 identity처럼 보일 수 있지만
fleet identity는 더 큰 개념이다.

예:

```text
Device ID
Certificate
Serial Number
Cloud Thing Name
Hardware UUID
```

등이 함께 사용될 수 있다.

---

# 7. 왜 고유 Identity가 중요한가?

예:

```text
Robot 001
→ Site A

Robot 002
→ Site B

Robot 003
→ Lab
```

각 robot에 다른 config나 permission을 적용할 수 있다.

---

# 8. Provisioning

새 robot을 fleet에 등록하는 과정을:

```text
Provisioning
```

이라고 한다.

보통 다음을 설정한다.

```text
Device Identity
Credential
Network Configuration
Software Version
Fleet Group
Cloud Endpoint
```

---

# 9. Factory Provisioning

Robot 생산 단계에서 미리:

```text
Certificate
Device ID
Initial software
```

를 넣을 수도 있다.

---

# 10. First-Boot Provisioning

Robot을 처음 켰을 때:

```text
Boot
  ↓
Registration
  ↓
Cloud/Fleet Join
```

형태로 자동 등록할 수도 있다.

---

# 11. Manual Provisioning

초기에는 사람이 직접:

```bash
ssh
copy certificate
configure service
```

하는 방식도 가능하다.

Robot 수가 적을 때는 단순하다.

---

# 12. Zero-Touch Provisioning

대규모 fleet에서는 사람이 robot마다 직접 설정하지 않는 것이 이상적이다.

```text
Power On
   ↓
Device authenticates
   ↓
Registers automatically
   ↓
Downloads configuration
   ↓
Starts service
```

이런 구조를:

```text
Zero-Touch Provisioning
```

이라고 볼 수 있다.

---

# 13. Credential

각 robot이 fleet service에 자신을 인증하기 위해:

```text
Certificate
Private Key
Token
```

등을 사용할 수 있다.

Chapter 16과 연결된다.

---

# 14. Per-Device Credential

좋은 구조:

```text
Robot A → Credential A
Robot B → Credential B
Robot C → Credential C
```

한 robot이 compromise되어도
전체 fleet credential이 유출되지 않는다.

---

# 15. Device Group

비슷한 robot을 group으로 묶을 수 있다.

예:

```text
Fleet
├── Vision60
├── Forklift
├── Blasting Robot
└── Test Robots
```

또는 site 기준:

```text
Site A
Site B
Lab
```

---

# 16. Group의 장점

Group 단위로:

```text
Software Deploy
Configuration
Monitoring
Permission
```

을 적용할 수 있다.

---

# 17. Software Version Management

각 robot이 어떤 version을 실행하는지 알아야 한다.

예:

```text
Robot 001 → autonomy v1.4.2
Robot 002 → autonomy v1.4.2
Robot 003 → autonomy v1.3.9
```

---

# 18. Version은 하나가 아닐 수 있다

Robot에는 여러 software가 존재한다.

예:

```text
OS Version
JetPack Version
Docker Image Version
Autonomy Version
SLAM Version
Config Version
Map Version
```

따라서 전체 deployment state를 기록하는 것이 중요하다.

---

# 19. Git Commit

Software version을 정확히 추적하려면:

```text
Git Commit
```

을 함께 기록하면 좋다.

예:

```text
Autonomy:
v1.4.2
commit: abc1234
```

---

# 20. Semantic Versioning

Version을:

```text
MAJOR.MINOR.PATCH
```

형태로 관리하는 방식을 많이 사용한다.

예:

```text
2.4.1
```

대략:

```text
MAJOR
→ 큰 호환성 변경

MINOR
→ 기능 추가

PATCH
→ 버그 수정
```

이다.

---

# 21. Configuration도 Versioning이 필요하다

Code가 같아도:

```text
FAST-LIO2 parameter
Camera config
Site map
Network config
```

가 다르면 robot 동작이 달라진다.

따라서:

```text
Code Version
+
Config Version
```

을 함께 관리한다.

---

# 22. Artifact

Build 결과물을:

```text
Artifact
```

라고 부른다.

예:

```text
Binary
Docker Image
Deb Package
Firmware
Config Bundle
```

---

# 23. Container Image를 Deployment Artifact로 사용

Robot software를 Docker image로 관리하면:

```text
vision60-autonomy:v1.4.2
```

같은 artifact를 여러 robot에 동일하게 배포할 수 있다.

---

# 24. Container Registry

Docker image를 저장하는 중앙 server:

```text
Container Registry
```

이다.

예:

```text
GitHub Container Registry
AWS ECR
Private Registry
```

---

# 25. Registry Architecture

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

# 26. 왜 Git Repository만으로 부족할까?

Robot에서 매번:

```bash
git clone
colcon build
pip install
```

하면 robot마다 결과가 달라질 수 있다.

예:

```text
Dependency version mismatch
Build environment mismatch
Build failure
```

---

# 27. Pre-Built Artifact

더 안정적인 방식:

```text
CI Server에서 build
      ↓
Test
      ↓
Artifact 생성
      ↓
Robot은 artifact만 다운로드
```

이다.

---

# 28. CI

CI:

```text
Continuous Integration
```

이다.

Code 변경이 생길 때 자동으로:

```text
Build
Test
Lint
Package
```

등을 수행한다.

---

# 29. CD

CD는 문맥에 따라:

```text
Continuous Delivery
Continuous Deployment
```

를 의미할 수 있다.

Robot software에서는 build된 artifact를
배포 가능한 상태로 만드는 과정과 연결된다.

---

# 30. CI/CD Pipeline

예:

```text
Developer Push
      │
      ▼
Git Repository
      │
      ▼
CI
├── Build
├── Test
├── Static Analysis
└── Package
      │
      ▼
Container Registry
      │
      ▼
Deployment System
      │
      ▼
Robot Fleet
```

---

# 31. ARM64 Build

Jetson은 ARM64이므로 artifact도 target architecture가 맞아야 한다.

예:

```text
linux/arm64
```

container image가 필요할 수 있다.

---

# 32. Multi-Architecture CI

개발 PC가 x86_64이고 robot은 ARM64라면:

```text
CI
├── linux/amd64
└── linux/arm64
```

이미지를 모두 만들 수도 있다.

---

# 33. Cross-Build

x86 server에서 ARM64 artifact를 만드는:

```text
Cross Compilation
Cross Build
```

을 사용할 수도 있다.

Chapter 2와 연결된다.

---

# 34. OTA Update

OTA:

```text
Over-The-Air Update
```

이다.

Network를 통해 robot software를 원격으로 업데이트하는 것.

---

# 35. OTA 기본 구조

```text
Deployment Server
       │
       ▼
    Internet /
 Company Network
       │
       ▼
      Robot
       │
       ▼
Download
       │
       ▼
Install / Switch
```

---

# 36. OTA는 단순 `git pull`이 아니다

실무적인 OTA system은 보통:

```text
Version Check
Authentication
Download
Integrity Check
Install
Restart
Health Check
Rollback
```

까지 포함한다.

---

# 37. Update Package 검증

Robot이 artifact를 받으면:

```text
Hash
Signature
```

를 검증할 수 있다.

Chapter 16의 signed update와 연결된다.

---

# 38. Deployment Target

Update를 모든 robot에 보낼 필요는 없다.

예:

```text
All Vision60
Site A robots
Test group
Robot 007 only
```

처럼 target을 선택할 수 있다.

---

# 39. Staged Rollout

새 version을 처음부터 모든 robot에 배포하지 않는다.

예:

```text
Stage 1
1 test robot

Stage 2
5% fleet

Stage 3
25%

Stage 4
100%
```

---

# 40. 왜 Staged Rollout을 할까?

Lab에서는 발견하지 못한 bug가
실제 field에서 발생할 수 있다.

```text
New Version
     ↓
One Robot
     ↓
Issue?
```

먼저 작은 group에서 확인하면 피해를 줄일 수 있다.

---

# 41. Canary Deployment

아주 작은 일부 device에 먼저 새 version을 배포하는 방법을:

```text
Canary Deployment
```

라고 부른다.

---

# 42. Deployment Ring

Fleet을 여러 ring으로 나눌 수도 있다.

```text
Ring 0
Developers

Ring 1
Lab Robots

Ring 2
Pilot Site

Ring 3
Production Fleet
```

---

# 43. Rollback

새 version에 문제가 생기면 이전 version으로 돌아가는 기능:

```text
Rollback
```

이다.

---

# 44. Container Rollback

예:

```text
Current:
vision60-autonomy:v1.4.2

Previous:
vision60-autonomy:v1.4.1
```

새 container가 실패하면 이전 image를 다시 실행한다.

---

# 45. Rollback이 어려운 경우

Software만 이전 version으로 돌리면 끝나지 않을 수 있다.

예:

```text
Config Format Changed
Database Migrated
Firmware Changed
Map Format Changed
```

등.

그래서 backward compatibility를 고려해야 한다.

---

# 46. A/B Partition

OS update에서는:

```text
Partition A
Partition B
```

두 system slot을 유지하는 방식을 사용할 수 있다.

예:

```text
Current Boot
→ A

Update
→ B에 설치

Reboot
→ B

Failure
→ A로 복귀
```

---

# 47. Atomic Update

Update 중 system이 반쯤만 변경되는 것을 피하고:

```text
Old Version
   ↓
One Switch
   ↓
New Version
```

처럼 전환하는 개념을:

```text
Atomic Update
```

라고 볼 수 있다.

---

# 48. Partial Update의 위험

예:

```text
FAST-LIO2 new
Nav2 old
Config new
```

처럼 version이 섞이면
호환성 문제가 생길 수 있다.

가능하면 compatible set을 하나의 deployment unit으로 관리한다.

---

# 49. Dependency Version Matrix

예:

| Component | Version |
|---|---|
| JetPack | 6.x |
| ROS 2 | Humble |
| Autonomy | 1.4.2 |
| FAST-LIO2 | commit abc |
| CycloneDDS Config | 3 |
| Site Config | 12 |

이런 식으로 compatibility를 기록할 수 있다.

---

# 50. Remote Command

Fleet system에서는 robot에 원격 command를 보낼 수도 있다.

예:

```text
Restart service
Start diagnostics
Download logs
Update config
Reboot
```

---

# 51. Remote Command는 매우 민감하다

Command channel이 compromise되면 공격자가 robot을 제어할 수 있다.

따라서:

```text
Authentication
Authorization
Encryption
Audit Log
```

이 매우 중요하다.

---

# 52. Control Plane과 Data Plane

Fleet architecture에서는 두 개념을 구분하면 좋다.

```text
Control Plane
→ Update / Configuration / Command

Data Plane
→ Robot operational data
```

---

# 53. Telemetry

Robot의 상태를 중앙으로 보내는 데이터를:

```text
Telemetry
```

라고 한다.

예:

```text
CPU Usage
GPU Usage
Battery
Temperature
Robot Pose
Current Mode
Sensor Health
Software Version
```

---

# 54. Telemetry vs Sensor Raw Data

Telemetry는 일반적으로 작고 요약된 상태 정보다.

```text
Telemetry
→ CPU 60%
→ Temp 70°C
→ LiDAR OK
```

반면:

```text
Point Cloud
Camera Video
```

는 훨씬 큰 raw sensor data다.

---

# 55. 왜 Telemetry를 따로 보낼까?

Robot 100대에서 모든 camera와 LiDAR raw data를 cloud로 보내면:

```text
Huge Bandwidth
Huge Storage
Huge Cost
```

가 필요하다.

그래서:

```text
Small Telemetry
```

를 계속 보내고,
raw data는 필요할 때만 가져오는 구조가 효율적일 수 있다.

---

# 56. Edge vs Cloud 역할 분리

```text
Robot Edge

├── Real-time control
├── SLAM
├── Perception
├── Local decisions
└── Raw sensor processing

Cloud

├── Fleet monitoring
├── Long-term analytics
├── Software deployment
├── Dataset storage
└── Fleet dashboard
```

---

# 57. Network가 끊겨도 Robot은 동작해야 할 수 있다

Field robot은 항상 인터넷이 연결된다고 가정하면 안 된다.

```text
Cloud
   X
Robot
```

이어도:

```text
Local Control
Local SLAM
Safety
```

는 계속 동작해야 할 수 있다.

---

# 58. Offline-First

Network가 없어도 핵심 기능은 local에서 동작하고,
network가 돌아오면 data를 sync하는 구조:

```text
Offline-First
```

로 설계할 수 있다.

---

# 59. Store-and-Forward

Network가 끊겼을 때 telemetry를 local에 저장한다.

```text
Network Offline
      ↓
Local Queue
      ↓
Network Restored
      ↓
Upload
```

이를:

```text
Store-and-Forward
```

라고 볼 수 있다.

---

# 60. Queue Size가 무한하면 안 된다

Network가 일주일 끊기면 local telemetry가 계속 쌓일 수 있다.

따라서:

```text
Queue limit
Retention policy
Drop policy
```

가 필요하다.

Chapter 13과 연결된다.

---

# 61. Device Shadow / Desired State

Fleet system에서는 중앙에서:

```text
Desired State
```

를 정의하고 robot이 현재 상태를 맞추도록 할 수 있다.

예:

```text
Desired:
software_version = 1.4.2

Reported:
software_version = 1.4.1
```

그러면 update가 필요하다는 것을 알 수 있다.

---

# 62. Desired vs Reported State

```text
Desired
→ 중앙에서 원하는 상태

Reported
→ Robot이 현재 보고하는 상태
```

이다.

---

# 63. Configuration Management

Software binary뿐 아니라:

```text
ROS_DOMAIN_ID
Site
Sensor Config
Power Mode
Logging Policy
```

같은 configuration을 중앙에서 관리할 수 있다.

---

# 64. Config Drift

같은 fleet인데 robot마다 config가 조금씩 달라지는 현상을:

```text
Configuration Drift
```

라고 할 수 있다.

예:

```text
Robot A
ROS_DOMAIN_ID=123

Robot B
ROS_DOMAIN_ID=10
```

문제가 생기기 쉽다.

---

# 65. Desired State Management

중앙에서:

```text
All Vision60:
ROS_DOMAIN_ID=123
Autonomy=v1.4.2
```

같이 desired state를 정의하면 drift를 줄일 수 있다.

---

# 66. Infrastructure as Code 개념

Configuration을 GUI에서 손으로만 바꾸는 대신
code/config file로 관리할 수 있다.

예:

```text
YAML
JSON
Ansible
Terraform
```

등.

---

# 67. Ansible

여러 Linux machine에:

```text
Package 설치
Config 복사
Service restart
```

같은 작업을 자동화할 수 있는 tool이다.

Robot 수가 적은 초기 fleet에서도 유용할 수 있다.

---

# 68. Ansible vs OTA Platform

Ansible:

```text
General server configuration automation
```

OTA/Fleet Platform:

```text
Device identity
Offline handling
Update status
Rollback
Fleet scale
```

등을 더 전문적으로 다룰 수 있다.

---

# 69. Fleet Health

각 robot에:

```text
Healthy
Warning
Critical
Offline
```

같은 상태를 정의할 수 있다.

---

# 70. Health Metric

예:

```text
CPU < 90%
Disk < 85%
Temperature < limit
LiDAR heartbeat present
Localization valid
```

등.

---

# 71. Heartbeat

Robot이 주기적으로:

```text
"I'm alive."
```

를 보내는 signal이다.

예:

```text
Robot
   │
   │ heartbeat every 30 s
   ▼
Fleet Server
```

---

# 72. Heartbeat가 없으면

즉시 robot 고장이라고 단정할 수 없다.

원인:

```text
Robot powered off
Network down
Cloud down
Process crashed
Credential expired
```

등 다양하다.

---

# 73. Online vs Healthy

```text
Online
```

은 network connection이 있다는 의미일 수 있다.

하지만:

```text
Healthy
```

는 sensor/algorithm까지 정상이라는 의미다.

둘은 다르다.

---

# 74. Example

```text
Robot online
SSH works

but

LiDAR dead
FAST-LIO2 stopped
```

이면:

```text
Online
but Unhealthy
```

이다.

---

# 75. Deployment Health Check

Update 후 자동으로 확인한다.

예:

```text
Container running?
      ↓
ROS node alive?
      ↓
Sensor topics healthy?
      ↓
Odometry publishing?
      ↓
Temperature normal?
```

---

# 76. 단순 Process Check만으로 부족하다

```text
FAST-LIO2 process exists
```

여도:

```text
/odometry = 0 Hz
```

일 수 있다.

그래서 application-level health check가 중요하다.

---

# 77. Deployment Timeout

Update 후 일정 시간 안에 healthy 상태가 되지 않으면:

```text
Deployment Failed
```

로 판단할 수 있다.

---

# 78. Automatic Rollback

예:

```text
Deploy v1.5
    ↓
Health Check Failed
    ↓
Rollback v1.4
```

자동화할 수 있다.

---

# 79. Telemetry Frequency

모든 metric을 초당 100번 cloud로 보낼 필요는 없다.

예:

```text
CPU
1 Hz

Software Version
on change

Battery
1 Hz

Critical Error
immediate
```

처럼 중요도에 따라 frequency를 정한다.

---

# 80. Event

주기적인 telemetry 외에 사건 발생 시 message를 보낼 수 있다.

예:

```text
E-stop pressed
Localization lost
Overheat
Disk almost full
Deployment failed
```

---

# 81. Command vs Event vs Telemetry

```text
Command
Cloud → Robot

Telemetry
Robot → Cloud, periodic

Event
Robot → Cloud, something happened
```

이다.

---

# 82. Bidirectional Communication

Fleet system은 양방향 통신이 필요할 수 있다.

```text
Cloud
   │
   ├──► Command
   │
   ◄── Telemetry
   ◄── Event
   │
Robot
```

---

# 83. MQTT

IoT fleet에서 자주 사용하는 protocol:

```text
MQTT
```

이다.

Publish/Subscribe 구조를 사용한다.

---

# 84. MQTT Structure

```text
Robot
 Publisher
     │
     ▼
MQTT Broker
     │
     ▼
Cloud Subscriber
```

반대로 command:

```text
Cloud Publisher
     │
     ▼
MQTT Broker
     │
     ▼
Robot Subscriber
```

---

# 85. Topic

MQTT에도:

```text
Topic
```

이라는 용어가 있다.

예:

```text
robots/vision60-001/telemetry
robots/vision60-001/commands
```

ROS topic과 이름은 같지만 완전히 다른 middleware 개념이다.

---

# 86. MQTT vs ROS 2

```text
ROS 2
→ Robot 내부 / local distributed robotics communication

MQTT
→ Cloud/IoT messaging에서 많이 사용
```

둘을 함께 사용할 수도 있다.

---

# 87. Example

```text
Robot Internal

LiDAR
 ↓
ROS 2
 ↓
FAST-LIO2
 ↓
Health Node
 ↓
MQTT
 ↓
Cloud
```

---

# 88. Raw ROS Topic을 그대로 Cloud에 보내는 것은?

가능할 수 있지만 항상 좋은 설계는 아니다.

예:

```text
PointCloud2
Camera Image
```

는 bandwidth가 매우 크다.

Cloud에는 필요한 summary만 보내는 것이 효율적일 수 있다.

---

# 89. Edge Aggregation

Robot 내부에서 여러 metric을 모아서
작은 telemetry message로 만든다.

```text
CPU
GPU
Battery
SLAM
Sensor
      │
      ▼
Telemetry Agent
      │
      ▼
Cloud
```

---

# 90. Greengrass

AWS IoT Greengrass 같은 edge runtime은
device에서 cloud-connected component를 실행하고 관리하는 데 사용할 수 있다.

개념:

```text
Cloud
  │
  ▼
Greengrass Deployment
  │
  ▼
Edge Device
  │
  ├── Component A
  ├── Component B
  └── Component C
```

---

# 91. Greengrass Core Device

Greengrass를 실행하는 edge machine을:

```text
Core Device
```

라고 부른다.

예:

```text
One Jetson
=
One Core Device
```

구조로 사용할 수 있다.

---

# 92. Thing

AWS IoT에서 physical/logical device를:

```text
Thing
```

으로 표현할 수 있다.

예:

```text
vision60-001
```

---

# 93. Thing Group

여러 Thing을 group으로 묶는다.

```text
Vision60Fleet
├── vision60-001
├── vision60-002
└── vision60-003
```

Group 단위 deployment에 사용할 수 있다.

---

# 94. Component

Greengrass에서 배포 가능한 software 단위를:

```text
Component
```

라고 부른다.

예:

```text
Telemetry Agent
Video Uploader
Robot Updater
Diagnostics
```

---

# 95. Component와 Docker Container

Component가 반드시 Docker container인 것은 아니다.

Component recipe가:

```text
Script
Binary
Docker
Artifact
```

등을 실행하도록 구성될 수 있다.

---

# 96. Deployment

Cloud에서:

```text
Thing
Thing Group
```

에 특정 component version을 적용하는 것을 deployment라고 볼 수 있다.

---

# 97. Robot Software 전체를 Cloud Runtime에 맡겨야 할까?

반드시 그런 것은 아니다.

예를 들어:

```text
Hard/Local Autonomy
→ ROS 2 / systemd / Docker

Cloud Agent
→ Greengrass
```

처럼 역할을 분리할 수 있다.

---

# 98. Cloud Dependency를 줄인다

Robot의 핵심 autonomy는:

```text
Internet down
```

이어도 계속 동작해야 할 수 있다.

따라서:

```text
Cloud = Management
Edge = Operation
```

구조가 중요하다.

---

# 99. Edge Agent

Robot마다 작은 management agent를 둘 수 있다.

역할:

```text
Check desired version
Download artifact
Verify artifact
Restart service
Report status
Upload logs
```

---

# 100. Agent가 너무 강한 권한을 가지면?

Security risk가 된다.

예:

```text
Management Agent
root access
cloud command
```

가 compromise되면 system 전체가 위험하다.

Chapter 16의 Least Privilege가 중요하다.

---

# 101. Deployment Security

Deployment channel에는:

```text
Authentication
TLS
Artifact Signature
Authorization
```

이 필요하다.

---

# 102. Artifact Integrity

Robot이 받은 image/package가
CI에서 만든 것과 같은지 검증한다.

예:

```text
Digest
Hash
Signature
```

---

# 103. Docker Digest

Docker image에는 content digest가 있다.

예:

```text
sha256:...
```

Tag는 변경될 수도 있지만
digest는 content를 명확하게 식별한다.

---

# 104. Tag vs Digest

```text
Tag
v1.4.2
→ 사람이 읽기 쉬움
→ 다른 image를 가리키도록 변경 가능

Digest
sha256:...
→ 정확한 content 식별
```

Production에서는 digest까지 기록하면 reproducibility가 더 좋아진다.

---

# 105. Deployment Manifest

어떤 robot에 무엇을 배포할지 하나의 manifest로 기록할 수 있다.

예:

```yaml
robot: vision60-001
autonomy: 1.4.2
config: site-a-v7
map: site-a-20260905
```

---

# 106. Fleet Database

Fleet server는 각 robot에 대해 다음 정보를 저장할 수 있다.

```text
Device ID
Serial Number
Software Version
Config Version
Last Seen
Health
Deployment Status
Site
```

---

# 107. Site Management

Robot이 여러 현장에 있다면:

```text
Shipyard A
Construction Site B
Lab
```

site metadata를 관리할 수 있다.

Site별 configuration도 다를 수 있다.

---

# 108. Environment-Specific Config

예:

```text
Site A
LiDAR map A

Site B
LiDAR map B
```

Software는 같고 config/map만 다르게 배포할 수 있다.

---

# 109. Fleet Drift

시간이 지나면서:

```text
Robot A
new config

Robot B
old config

Robot C
manual hotfix
```

처럼 상태가 달라질 수 있다.

이를 줄이는 것이 fleet management의 중요한 역할이다.

---

# 110. Manual Hotfix의 문제

SSH로 robot 하나에 직접 수정:

```bash
vim config.yaml
```

하면 중앙 system은 이 변경을 모를 수 있다.

나중에:

```text
"왜 Robot A만 다르지?"
```

문제가 생긴다.

---

# 111. Declarative Management

더 좋은 방식은:

```text
Desired Configuration
```

을 중앙에 정의하고
robot이 그 상태와 맞도록 관리하는 것이다.

---

# 112. Immutable Deployment

Production에서는 running robot 내부 source를 직접 수정하기보다:

```text
New Artifact
   ↓
New Version
   ↓
Deploy
```

하는 방식이 관리하기 쉽다.

---

# 113. Roll Forward

문제 발생 시 이전 version으로 돌아가는 rollback 외에도
새 patch version을 빠르게 만들어:

```text
v1.4.2
 ↓ bug
v1.4.3
```

로 해결할 수 있다.

이를:

```text
Roll Forward
```

전략으로 볼 수 있다.

---

# 114. Deployment Window

Robot이 중요한 작업 중일 때 update하면 위험할 수 있다.

예:

```text
Robot moving
      ↓
Software restart
```

그래서 update 가능한 시간대를 정할 수 있다.

---

# 115. Maintenance Mode

Update 전에:

```text
Stop autonomy
Park robot
Save logs
Enter maintenance
Update
Verify
Resume
```

같은 workflow를 만들 수 있다.

---

# 116. Safe Deployment

Robot software update는 web server update보다 더 신중해야 한다.

왜냐하면:

```text
Software
   ↓
Physical Machine
   ↓
Motion
```

으로 연결되기 때문이다.

---

# 117. Pre-Deployment Check

예:

```text
Robot stationary?
Battery sufficient?
Network stable?
Disk sufficient?
No critical task running?
```

를 확인할 수 있다.

---

# 118. Battery와 OTA

Update 중 battery가 꺼지면:

```text
Corrupted installation
```

가능성이 있다.

따라서 update 전 battery threshold를 확인하는 것이 좋다.

---

# 119. Network Resume

Artifact가 큰 경우 download 중 network가 끊길 수 있다.

좋은 updater는:

```text
Resume
Retry
Checksum
```

을 지원하는 것이 좋다.

---

# 120. Exponential Backoff

Cloud connection이 실패할 때 매 millisecond마다 재시도하면 network/server를 과부하시킬 수 있다.

그래서:

```text
1 s
2 s
4 s
8 s
...
```

처럼 재시도 간격을 늘릴 수 있다.

이를:

```text
Exponential Backoff
```

라고 한다.

---

# 121. Idempotency

같은 deployment command가 여러 번 와도
결과가 망가지지 않도록 설계하는 특성:

```text
Idempotency
```

이다.

예:

```text
"ensure autonomy v1.4.2 is running"
```

command를 두 번 받아도 최종 상태는 같다.

---

# 122. Command ID

Remote command에 unique ID를 붙일 수 있다.

```text
cmd-12345
```

Robot이 이미 처리한 command인지 확인해
duplicate execution을 방지할 수 있다.

---

# 123. State Machine

Deployment 자체를 state machine으로 관리하면 좋다.

예:

```text
IDLE
 ↓
DOWNLOADING
 ↓
VERIFYING
 ↓
INSTALLING
 ↓
RESTARTING
 ↓
HEALTH_CHECK
 ↓
SUCCESS
```

실패:

```text
FAILED
 ↓
ROLLBACK
```

---

# 124. Fleet Deployment Status

중앙 dashboard에서:

```text
20 Successful
2 Updating
1 Failed
3 Offline
```

같이 볼 수 있다.

---

# 125. Offline Robot

Robot이 offline이면 즉시 update할 수 없다.

중앙에는:

```text
Desired Version = 1.4.2
```

를 저장해두고
robot이 reconnect되면 update할 수 있다.

---

# 126. Eventual Consistency

모든 robot이 동시에 즉시 같은 상태가 되지 않더라도
시간이 지나며 desired state에 수렴하는 구조를:

```text
Eventual Consistency
```

관점으로 볼 수 있다.

---

# 127. Fleet Management와 Observability

Fleet management가:

```text
무엇을 배포할까?
```

라면 observability는:

```text
지금 system이 어떻게 동작하고 있는가?
```

를 보는 것이다.

Chapter 18에서 본격적으로 다룬다.

---

# 128. Data Flow Example

```text
                   Cloud

          ┌─────────────────────┐
          │ Fleet Manager       │
          │                     │
          │ Desired Version     │
          │ Config              │
          │ Deployment          │
          └──────────┬──────────┘
                     │
                   TLS
                     │
                     ▼
                 Robot Agent
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Docker      Config     Logs
          │
          ▼
       ROS 2
          │
          ▼
      Autonomy
```

---

# 129. Vision60 Fleet Example

```text
                     Vision60 Fleet

              ┌────────────────────┐
              │ Fleet Management   │
              └─────────┬──────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
  Vision60-001    Vision60-002    Vision60-003
      │               │               │
      ▼               ▼               ▼
    Orin             Orin            Orin
      │               │               │
      ├── Docker      ├── Docker      ├── Docker
      ├── ROS 2       ├── ROS 2       ├── ROS 2
      └── FAST-LIO2   └── FAST-LIO2   └── FAST-LIO2
```

---

# 130. Fleet Update Example

```text
Autonomy v1.5.0
      │
      ▼
CI Test
      │
      ▼
ARM64 Image
      │
      ▼
Registry
      │
      ▼
Test Robot
      │
      ▼
Lab Fleet
      │
      ▼
Production Fleet
```

---

# 131. Failure Example

```text
Deploy v1.5.0
      │
      ▼
FAST-LIO2 fails to publish
      │
      ▼
Health Check Failure
      │
      ▼
Stop rollout
      │
      ▼
Rollback test robots
```

중앙 rollout system이 있으면
문제가 전체 fleet으로 확산되는 것을 막을 수 있다.

---

# 132. Remote Debugging

Fleet system에서 robot의:

```text
Logs
Metrics
Version
Network Status
```

를 원격으로 확인할 수 있다.

하지만 무조건 remote shell access를 모든 사람에게 주는 것보다
필요한 diagnostic 기능을 별도로 제공하는 것이 더 안전할 수 있다.

---

# 133. Log Upload

Robot이 문제가 생기면:

```text
Selected logs
Crash dump
Small rosbag
Diagnostics
```

를 중앙 server에 업로드할 수 있다.

---

# 134. Raw Data Upload 정책

모든 raw sensor data를 항상 올리는 대신:

```text
On demand
Failure event
Wi-Fi available
Robot charging
```

같은 조건에서 upload할 수 있다.

---

# 135. Bandwidth-Aware Upload

Robot이 cellular network를 사용한다면 bandwidth 비용이 중요할 수 있다.

예:

```text
Telemetry
→ always

Logs
→ compressed

Large bag
→ Wi-Fi only
```

같은 정책을 사용할 수 있다.

---

# 136. Deployment와 Site Network

현장 network가 불안정할 수 있다.

따라서 deployment system은:

```text
Retry
Resume
Partial download handling
Offline mode
```

를 고려해야 한다.

---

# 137. Fleet Security Checklist

```text
[ ] Per-device identity
[ ] Unique credential
[ ] TLS
[ ] Artifact signature
[ ] Least privilege
[ ] Remote commands audited
[ ] Credential revocation
[ ] Rollback supported
```

---

# 138. Deployment Checklist

```text
[ ] Target robots correct?
[ ] Correct architecture?
[ ] Version fixed?
[ ] Artifact verified?
[ ] Config compatible?
[ ] Battery sufficient?
[ ] Disk sufficient?
[ ] Rollback available?
[ ] Health check defined?
[ ] Staged rollout?
```

---

# 139. Mini Practice 1

Vision60이 3대 있다고 가정한다.

다음 정보를 정의한다.

| Robot | Device ID | Site | Software | Config |
|---|---|---|---|---|
| Robot 1 | vision60-001 | Lab | v1.4.2 | lab-v3 |
| Robot 2 | vision60-002 | Site A | v1.4.2 | site-a-v7 |
| Robot 3 | vision60-003 | Site A | v1.4.1 | site-a-v7 |

질문:

```text
어느 robot이 version drift 상태인가?
```

---

# 140. Mini Practice 2

새 version:

```text
v1.5.0
```

을 배포한다고 하자.

직접 rollout plan을 작성한다.

예:

```text
Step 1
Lab robot 1대

Step 2
1 hour health monitoring

Step 3
Site A 1대

Step 4
Remaining production robots
```

---

# 141. Mini Practice 3

Update health check를 정의한다.

예:

```text
Container running
ROS graph available
LiDAR > 9 Hz
IMU > 180 Hz
Odometry > required rate
Temperature below limit
```

---

# 142. Mini Practice 4

Network가 끊겼다고 하자.

질문:

```text
Robot autonomy는 계속 동작해야 하는가?

Telemetry는 어디에 저장할까?

Reconnect 후 어떤 data를 먼저 upload할까?
```

---

# 143. Mini Practice 5

Fleet command:

```text
Restart FAST-LIO2
```

를 설계한다고 하자.

필요한 것:

```text
Authentication
Authorization
Command ID
Audit Log
Result Status
Timeout
```

을 생각한다.

---

# 144. Mini Practice 6

현재 robot update 방식이:

```text
SSH
git pull
manual restart
```

라고 가정한다.

이를:

```text
Versioned Artifact
Registry
Deployment Agent
Health Check
Rollback
```

구조로 바꾸면 어떤 장점이 있는지 설명한다.

---

# 145. 반드시 구분할 것

```text
Fleet
≠
Network

Provisioning
≠
Deployment

Telemetry
≠
Raw Sensor Data

Command
≠
Telemetry

Online
≠
Healthy

Tag
≠
Digest

Update
≠
Rollback

CI
≠
CD

Robot Group
≠
ROS_DOMAIN_ID

Cloud Management
≠
Robot Real-Time Control
```

---

# 146. Fleet Mental Model

```text
Source Code
    │
    ▼
Git
    │
    ▼
CI
    │
    ▼
Test
    │
    ▼
Artifact
    │
    ▼
Registry
    │
    ▼
Fleet Manager
    │
    ▼
Deployment
    │
    ▼
Robot
    │
    ▼
Health Check
    │
    ├── Success
    │
    └── Rollback
```

---

# 147. Edge/Cloud Mental Model

```text
                       Cloud

             Fleet Management
             Deployment
             Analytics
             Long-term Storage
             Dashboard
                    │
                    │ Secure Network
                    ▼
                       Edge

                Jetson / Robot
                ├── ROS 2
                ├── SLAM
                ├── AI
                ├── Navigation
                └── Safety
```

핵심 robot operation은 edge에서 실행하고,
cloud는 management와 fleet-scale operation을 담당한다.

---

# 148. Chapter 17 핵심

Robot fleet management의 핵심 질문은:

```text
What software is running where?

How do we update it safely?

How do we know the update worked?

How do we recover when it fails?

How do we manage hundreds of devices consistently?
```

이다.

한 대의 robot에서는:

```text
SSH
```

만으로 충분할 수 있지만,

Fleet에서는:

```text
Identity
Provisioning
Versioning
CI/CD
OTA
Staged Rollout
Health Check
Rollback
Telemetry
```

가 필요해진다.

---

# Next Chapter

## Chapter 18. Observability & Monitoring

다음 Chapter에서는:

```text
Logs
Metrics
Tracing
Dashboard
Alert
Health Check
CPU/GPU/RAM
Temperature
Disk
Network
ROS Topic Rate
Application Latency
```

를 다룬다.

핵심 질문은:

```text
"Robot이 지금 살아 있는가?"
```

보다 한 단계 더 깊은:

```text
"Robot 내부에서 지금 무슨 일이 일어나고 있는가?"
```

이다.

특히 다음 구조를 다룬다.

```text
Robot
  │
  ├── Logs
  ├── Metrics
  ├── Events
  └── Traces
       │
       ▼
Monitoring System
       │
       ▼
Dashboard / Alert
```

Chapter 18까지 이해하면 remote fleet에서 문제가 생겼을 때
무조건 SSH로 들어가서 확인하지 않고도 원인을 빠르게 좁히는 구조를 이해할 수 있다.