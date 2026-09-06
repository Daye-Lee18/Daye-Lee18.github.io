---
title: "Chapter 16. Embedded Security"
importance: 17
---

> **Goal:** Jetson과 robot edge computer를 network에 연결할 때 필요한 기본 security 개념을 이해한다.
>
> User/Root, SSH Key, Firewall, Port, Least Privilege, Secret Management,
> TLS, Certificate, Secure Boot, Disk Encryption, Container Security,
> Software Update의 기본을 이해하고 실제 robot deployment에 연결한다.

---

# 1. Robot Security는 왜 중요한가?

로봇이 network에 연결되면 단순한 embedded device가 아니라
networked computer가 된다.

예:

```text
Robot
  │
  ├── Jetson
  ├── Xavier
  ├── LiDAR
  ├── Camera
  ├── MCU
  └── Wi-Fi / Ethernet
```

외부에서는:

```text
Laptop
Company Network
Cloud
Internet
```

와 연결될 수 있다.

즉 공격 surface가 커진다.

---

# 2. Attack Surface

Attack Surface는:

> 공격자가 접근하거나 악용할 수 있는 모든 접점

이다.

예:

```text
SSH
Open Ports
Web UI
Wi-Fi
Bluetooth
USB
Cloud API
Container
Default Password
```

---

# 3. Robot은 일반 서버보다 더 위험할 수 있다

서버가 해킹되면:

```text
Data leak
Service failure
```

가 발생할 수 있다.

로봇은 추가로:

```text
Physical movement
Actuator control
Sensor manipulation
Safety risk
```

까지 이어질 수 있다.

---

# 4. Security의 기본 목표

보안의 핵심은 자주 다음 세 가지로 설명한다.

```text
Confidentiality
Integrity
Availability
```

---

# 5. Confidentiality

허가되지 않은 사람이 데이터를 보지 못하게 하는 것.

예:

```text
Camera footage
Map
Robot logs
Credentials
```

---

# 6. Integrity

데이터나 command가 임의로 변경되지 않도록 하는 것.

예:

```text
Navigation Command
Firmware
Software Update
Config
```

가 공격자에 의해 바뀌면 위험하다.

---

# 7. Availability

필요할 때 system을 정상적으로 사용할 수 있게 하는 것.

예:

```text
Network flooding
Disk full
Resource exhaustion
```

으로 robot service가 마비되면 availability 문제가 된다.

---

# 8. Authentication

Authentication은:

> 누구인지 확인하는 것

이다.

예:

```text
Username + Password
SSH Key
Certificate
```

---

# 9. Authorization

Authorization은:

> 인증된 사용자가 무엇을 할 수 있는지 결정하는 것

이다.

예:

```text
User A
→ log read only

User B
→ service restart

Admin
→ software update
```

---

# 10. Authentication vs Authorization

```text
Authentication
→ 당신은 누구인가?

Authorization
→ 무엇을 할 수 있는가?
```

이다.

---

# 11. Root

Linux의 root user는 매우 강력한 권한을 가진다.

Root는:

```text
System file 수정
User 관리
Network 설정
Driver/module 관리
Service 제어
```

등을 할 수 있다.

---

# 12. 일반 User와 Root 구분

일반적으로 application을 항상 root로 실행하는 것은 좋지 않다.

왜냐하면 application compromise가 발생했을 때:

```text
Application compromise
      ↓
Root privilege
      ↓
Entire system compromise
```

로 이어질 수 있기 때문이다.

---

# 13. Least Privilege

보안에서 매우 중요한 원칙:

```text
Least Privilege
```

이다.

의미:

> 필요한 최소한의 권한만 부여한다.

예:

```text
Camera Node
→ /dev/video0 access만 필요

SLAM Node
→ root 필요 없음
```

---

# 14. `sudo`

`sudo`는 필요할 때 일시적으로 높은 권한으로 command를 실행하는 방법이다.

예:

```bash
sudo systemctl restart robot.service
```

무조건 모든 command에 `sudo`를 붙이는 습관은 좋지 않다.

---

# 15. Default Password

Robot이나 embedded device에 기본 password가 설정되어 있는 경우가 있다.

예:

```text
user: admin
password: admin
```

이 상태로 network에 연결하면 매우 위험하다.

따라서 초기 setup에서:

```text
Default password 변경
```

이 중요하다.

---

# 16. SSH Security

SSH는 remote shell access를 제공한다.

예:

```bash
ssh user@robot
```

즉 SSH access를 얻은 사용자는 robot 내부에서 command를 실행할 수 있다.

---

# 17. SSH Password Authentication

간단한 방식:

```text
Username
+
Password
```

하지만 password는:

```text
Guessing
Reuse
Brute Force
Phishing
```

위험이 있다.

---

# 18. SSH Key

더 안전하고 편리한 방식으로:

```text
SSH Public Key Authentication
```

을 많이 사용한다.

구조:

```text
Laptop

Private Key
    │
    ▼

Robot

Public Key
```

---

# 19. Private Key

Private key는 절대 다른 사람에게 공개하면 안 된다.

예:

```text
~/.ssh/id_ed25519
```

이 private key를 가진 사람은
해당 key가 허용된 system에 접근할 수 있다.

---

# 20. Public Key

Public key는 remote machine에 등록한다.

예:

```text
~/.ssh/authorized_keys
```

---

# 21. SSH Key 생성

```bash
ssh-keygen
```

예:

```text
Private Key
id_ed25519

Public Key
id_ed25519.pub
```

---

# 22. Public Key 복사

```bash
ssh-copy-id user@robot-ip
```

또는 public key를 직접:

```text
~/.ssh/authorized_keys
```

에 추가할 수 있다.

---

# 23. SSH Config

여러 robot을 관리할 때:

```text
~/.ssh/config
```

를 사용할 수 있다.

예:

```text
Host vision60
    HostName 192.168.10.10
    User robot
    IdentityFile ~/.ssh/id_ed25519
```

그러면:

```bash
ssh vision60
```

처럼 사용할 수 있다.

---

# 24. SSH Private Key를 GitHub에 올리면 안 된다

절대 repository에 넣으면 안 되는 것:

```text
Private SSH Key
Password
API Token
AWS Access Key
Cloud Credential
Certificate Private Key
```

특히 public repository에는 매우 위험하다.

---

# 25. Secret

Password, token, private key 같은 민감한 인증 정보를:

```text
Secret
```

이라고 부른다.

예:

```text
AWS_ACCESS_KEY_ID
API_TOKEN
PRIVATE_KEY
```

---

# 26. Secret를 Code에 Hard-Code하지 않는다

나쁜 예:

```python
TOKEN = "abc123-secret-token"
```

이 code를 Git에 올리면 secret도 함께 유출된다.

---

# 27. Environment Variable

Secret을 environment variable로 전달할 수 있다.

예:

```bash
export API_TOKEN=...
```

하지만 environment variable도 완벽하게 안전한 storage는 아니다.

중요한 것은:

```text
Secret를 source code와 분리
```

하는 것이다.

---

# 28. `.env`

예:

```text
API_TOKEN=...
ROBOT_PASSWORD=...
```

를 `.env` file에 저장할 수 있다.

이 file은:

```text
.gitignore
```

에 넣어 Git에 올라가지 않도록 해야 한다.

---

# 29. Secret Manager

Production에서는 더 안전한 secret management system을 사용할 수 있다.

예:

```text
Cloud Secret Manager
Vault
Hardware-backed key storage
```

등.

---

# 30. Port

Network application은 port를 사용한다.

예:

```text
22
→ SSH

80
→ HTTP

443
→ HTTPS
```

---

# 31. Open Port

어떤 port에서 service가 외부 connection을 받고 있다면:

```text
Open Port
```

라고 볼 수 있다.

필요 없는 port가 열려 있으면 attack surface가 증가한다.

---

# 32. Listening Port 확인

Linux:

```bash
ss -tulpen
```

또는:

```bash
ss -tuln
```

으로 listening port를 확인할 수 있다.

---

# 33. Firewall

Firewall은:

> 어떤 network traffic을 허용하고 차단할지 결정

한다.

예:

```text
Allow SSH from company subnet
Block unknown inbound traffic
```

---

# 34. UFW

Ubuntu에서는:

```bash
ufw
```

를 사용할 수 있다.

상태:

```bash
sudo ufw status
```

---

# 35. Firewall의 목적

모든 traffic을 열어놓는 대신:

```text
Required Traffic Only
```

를 허용하는 것이 원칙이다.

---

# 36. Example

Robot에서 필요한 것이:

```text
SSH
ROS 2 internal traffic
Monitoring
```

뿐이라면 불필요한 web/database port는 열 필요가 없다.

---

# 37. Firewall을 무작정 켜면 ROS 2가 안 될 수 있다

ROS 2 DDS는 여러 port와 multicast를 사용할 수 있다.

그래서:

```text
Firewall enabled
      ↓
ROS 2 discovery blocked
```

가 발생할 수 있다.

즉 security와 connectivity를 함께 설계해야 한다.

---

# 38. Network Segmentation

Robot network와 company network를 분리할 수 있다.

예:

```text
Company Network
      │
      ▼
Gateway
      │
      ▼
Robot Network
```

Robot sensor network를 외부에 직접 노출하지 않는 것이 좋다.

---

# 39. Internal Network

예:

```text
Robot Internal LAN

Xavier
Orin
LiDAR
Camera
```

이 network는 외부 internet과 직접 연결하지 않을 수 있다.

---

# 40. External Network

예:

```text
Company Wi-Fi
Internet
Cloud
```

외부 communication용 network다.

---

# 41. Dual Interface Architecture

Jetson:

```text
eth0
→ Robot Internal Network

wlan0
→ Company Network
```

처럼 두 network를 사용할 수 있다.

---

# 42. 이런 구조의 보안 장점

Sensor와 low-level robot network를 외부와 분리할 수 있다.

```text
Internet
   │
   X
LiDAR / MCU
```

직접 접근을 막을 수 있다.

---

# 43. Routing 주의

Jetson이 두 network 사이에서 routing을 수행하면
원하지 않게 internal network가 외부에 노출될 수 있다.

따라서:

```text
IP forwarding
NAT
Firewall
```

설정을 신중히 해야 한다.

---

# 44. ROS 2 Security

ROS 2는 network에서 node끼리 통신한다.

기본 환경에서는 같은 domain/network에서
원하지 않는 node가 discovery될 가능성을 고려해야 한다.

---

# 45. SROS2

ROS 2에는 security extension으로:

```text
SROS2
```

가 있다.

DDS Security를 활용해:

```text
Authentication
Encryption
Access Control
```

을 구성할 수 있다.

---

# 46. DDS Security

DDS Security specification에는:

```text
Authentication
Access Control
Cryptographic Protection
```

기능이 있다.

ROS 2에서 SROS2를 통해 사용할 수 있다.

---

# 47. ROS_DOMAIN_ID는 Security 기능이 아니다

매우 중요하다.

```text
ROS_DOMAIN_ID
```

는 discovery domain을 분리하는 데 사용한다.

하지만:

```text
ROS_DOMAIN_ID
≠
Authentication
```

이다.

같은 domain ID를 알아내면 접근할 수 있는 환경도 있을 수 있다.

---

# 48. ROS_DOMAIN_ID = 방화벽이 아니다

즉:

```text
ROS_DOMAIN_ID=123
```

만 설정했다고:

```text
Secure Robot Network
```

가 되는 것은 아니다.

---

# 49. Encryption

Network에서 데이터를 암호화하면
중간에서 packet을 보더라도 내용을 쉽게 읽지 못하게 할 수 있다.

```text
Plain Data
    ↓
Encryption
    ↓
Ciphertext
```

---

# 50. TLS

TLS:

```text
Transport Layer Security
```

이다.

Network communication의:

```text
Encryption
Authentication
Integrity
```

를 제공하는 데 사용된다.

---

# 51. HTTPS

HTTPS는:

```text
HTTP
+
TLS
```

이다.

그래서 web browser에서:

```text
https://
```

를 사용하는 것이다.

---

# 52. Certificate

TLS에서는 certificate를 사용해
server/client identity를 확인할 수 있다.

Certificate에는 보통:

```text
Public Key
Identity Information
Signature
```

등이 포함된다.

---

# 53. X.509 Certificate

IoT/AWS 환경에서 자주 보는:

```text
X.509 Certificate
```

는 public key certificate 표준이다.

Robot device authentication에 사용할 수 있다.

---

# 54. Certificate와 Private Key

보통:

```text
Certificate
→ 공개 가능

Private Key
→ 비밀
```

이다.

Private key는 절대 외부에 노출되면 안 된다.

---

# 55. Mutual TLS

일반 TLS에서는 주로 client가 server를 인증한다.

Mutual TLS:

```text
mTLS
```

에서는:

```text
Client verifies Server
+
Server verifies Client
```

둘 다 certificate로 인증할 수 있다.

IoT device identity에서 많이 사용된다.

---

# 56. Robot Identity

Fleet system에서는 각 robot에:

```text
Unique Device Identity
```

를 부여하는 것이 좋다.

예:

```text
Robot 001
→ Certificate A

Robot 002
→ Certificate B
```

---

# 57. 한 Credential을 모든 Robot이 공유하면?

위험하다.

```text
Robot A compromised
      ↓
Shared Credential stolen
      ↓
Robot B/C/D도 접근 가능
```

이 될 수 있다.

---

# 58. Per-Device Credential

더 좋은 구조:

```text
Robot A → Key A
Robot B → Key B
Robot C → Key C
```

한 robot이 compromise되어도
해당 credential만 revoke할 수 있다.

---

# 59. Revocation

Credential이 유출되었거나 robot이 분실되면:

```text
Revoke
```

할 수 있어야 한다.

즉 더 이상 해당 credential을 신뢰하지 않는 것이다.

---

# 60. Secure Boot

Secure Boot는:

> 부팅 시 신뢰된 software만 실행되도록 검증

하는 기술이다.

구조:

```text
Power On
   ↓
Boot ROM
   ↓
Verify Bootloader
   ↓
Verify Kernel
   ↓
Boot
```

---

# 61. 왜 Secure Boot가 필요할까?

공격자가 storage를 수정해서
악성 kernel/bootloader를 넣는 것을 막는 데 도움이 된다.

---

# 62. Chain of Trust

각 단계가 다음 단계를 cryptographically verify한다.

```text
Root of Trust
      ↓
Bootloader
      ↓
Kernel
      ↓
OS
```

이를:

```text
Chain of Trust
```

라고 한다.

---

# 63. Secure Boot가 모든 공격을 막는가?

아니다.

Secure Boot는 주로:

```text
Boot integrity
```

를 보호한다.

Application vulnerability, stolen password 등은 별도 문제다.

---

# 64. Disk Encryption

Storage를 암호화하면
SSD를 분리해 다른 computer에서 읽더라도
내용을 보호할 수 있다.

예:

```text
Robot stolen
      ↓
NVMe removed
      ↓
Encrypted Data
```

---

# 65. Full Disk Encryption

전체 filesystem/storage를 암호화할 수 있다.

Linux에서는:

```text
LUKS
```

같은 기술을 사용할 수 있다.

---

# 66. Disk Encryption의 Trade-off

장점:

```text
Data confidentiality
```

단점:

```text
Key management
Boot complexity
Performance overhead
Recovery complexity
```

가 있다.

---

# 67. Physical Access

Embedded security에서 중요한 현실:

> 공격자가 hardware를 직접 만질 수 있다.

예:

```text
USB port
Debug port
SSD
Ethernet
Serial console
```

에 접근할 수 있다.

---

# 68. USB Security

Robot의 외부 USB port에 아무 device나 연결할 수 있다면
위험할 수 있다.

예:

```text
Malicious USB device
Storage device
Keyboard emulation
```

등.

필요하지 않은 physical port는 제한할 수 있다.

---

# 69. Debug Interface

개발 board에는:

```text
UART console
JTAG
Recovery mode
```

같은 debug interface가 있을 수 있다.

Production에서는 이런 interface가 공격 경로가 될 수 있다.

---

# 70. JTAG

JTAG는 hardware debugging에 사용되는 interface다.

CPU debugging이나 firmware programming 등에 사용할 수 있다.

강력한 access가 가능할 수 있으므로 production에서 보호가 필요하다.

---

# 71. Container Security

Docker container는 isolation을 제공하지만
완벽한 security boundary라고 가정하면 안 된다.

---

# 72. Root Container

Container를 root로 실행하면
container 내부에서 높은 권한을 가진다.

Host와 device를 많이 expose한 경우 위험이 커질 수 있다.

---

# 73. `--privileged`

다음 옵션:

```bash
--privileged
```

는 container에 매우 강한 system access를 준다.

편하지만 보안상 위험하다.

필요하지 않다면 피하는 것이 좋다.

---

# 74. Device 최소 노출

나쁜 방식:

```bash
--privileged
```

대신 필요한 경우:

```bash
--device=/dev/video0
```

처럼 필요한 device만 전달하는 것이 더 안전하다.

---

# 75. Read-Only Filesystem

Application이 runtime에 filesystem을 수정할 필요가 없다면
일부 filesystem을 read-only로 구성할 수도 있다.

변조 surface를 줄일 수 있다.

---

# 76. Container Image Security

Container image에:

```text
Old package
Known vulnerability
Secret
Debug tool
```

이 들어있을 수 있다.

따라서 image도 관리 대상이다.

---

# 77. Minimal Image

Production image에는 필요한 것만 넣는 것이 좋다.

```text
Compiler
Debug tools
Unused package
```

가 필요 없다면 제거할 수 있다.

Attack surface와 image size를 줄인다.

---

# 78. Dependency Vulnerability

사용하는 library 자체에 보안 취약점이 있을 수 있다.

예:

```text
OpenSSL
Python package
ROS dependency
System library
```

그래서 dependency version 관리와 security update가 필요하다.

---

# 79. 업데이트의 딜레마

보안상:

```text
Update regularly
```

가 중요하다.

하지만 robot에서는 update로:

```text
Driver broken
CUDA mismatch
ROS incompatibility
```

가 생길 수도 있다.

---

# 80. 무작정 `apt upgrade`가 위험한 이유

Jetson은:

```text
JetPack
Kernel
Driver
CUDA
```

dependency가 밀접하다.

무심코 system update를 하면 compatibility가 깨질 수 있다.

---

# 81. Controlled Update

더 좋은 방식:

```text
Test Environment
      ↓
Update
      ↓
Regression Test
      ↓
Deploy
```

이다.

---

# 82. Version Pinning

Production에서는:

```text
OS Version
JetPack Version
Docker Image
Application Commit
Config
```

를 명확히 기록한다.

---

# 83. OTA Update

OTA:

```text
Over-The-Air Update
```

이다.

Robot에 physical access 없이 network를 통해 software를 업데이트한다.

---

# 84. OTA 장점

Fleet이 커지면:

```text
10 robots
100 robots
1000 robots
```

각 robot에 직접 USB를 연결해 업데이트하기 어렵다.

OTA가 필요해진다.

---

# 85. OTA 위험

잘못된 update를 모든 robot에 동시에 배포하면:

```text
Fleet-wide failure
```

가 발생할 수 있다.

---

# 86. Staged Rollout

따라서:

```text
1 Robot
   ↓
Small Group
   ↓
Entire Fleet
```

순서로 배포하는 방법이 안전하다.

---

# 87. Rollback

Update 후 문제가 생기면 이전 version으로 돌아갈 수 있어야 한다.

```text
v2.0
 ↓ failure
v1.9
```

---

# 88. Signed Update

Software update file에 digital signature를 사용하면
신뢰된 update인지 검증할 수 있다.

```text
Update Package
      +
Signature
      ↓
Robot verifies
```

---

# 89. Code Signing

공격자가 악성 update를 배포하는 것을 막기 위해
software artifact를 cryptographically sign할 수 있다.

---

# 90. Integrity Check

Hash를 이용해 file이 변경되었는지 확인할 수도 있다.

예:

```bash
sha256sum file
```

하지만 hash만으로는 누가 만든 file인지 인증하지는 못한다.

---

# 91. Hash vs Signature

```text
Hash
→ 내용이 같은가?

Digital Signature
→ 내용이 변하지 않았는가?
  +
  신뢰된 주체가 서명했는가?
```

이다.

---

# 92. Audit Log

누가 언제 어떤 작업을 했는지 기록하면
보안 사고 분석에 도움이 된다.

예:

```text
User A SSH login
User B software update
Robot config changed
```

---

# 93. Login 기록

Linux에서는:

```bash
last
```

같은 command로 login history 일부를 확인할 수 있다.

환경에 따라 system journal도 확인할 수 있다.

---

# 94. Security Log

SSH 관련:

```bash
journalctl -u ssh
```

또는 system 환경에 따라:

```text
/var/log/auth.log
```

등을 확인할 수 있다.

---

# 95. Brute Force

공격자가 많은 password를 반복해서 시도하는 것을:

```text
Brute Force Attack
```

이라고 한다.

---

# 96. SSH를 Internet에 바로 열지 않는다

가능하다면:

```text
Internet
   │
   X
Robot SSH
```

직접 노출을 피한다.

대신:

```text
VPN
Bastion
Private Network
```

등을 사용할 수 있다.

---

# 97. VPN

VPN:

```text
Virtual Private Network
```

이다.

Internet 위에 암호화된 private network처럼 사용할 수 있다.

```text
Laptop
   │
Internet
   │
Encrypted Tunnel
   │
VPN
   │
Robot Network
```

---

# 98. Bastion Host

내부 robot network에 직접 접속하지 않고
중간 관리 server를 통해 접근할 수도 있다.

```text
Developer
   │
   ▼
Bastion
   │
   ▼
Robot
```

---

# 99. Zero Trust 개념

단순히:

```text
"회사 network 안이니까 안전하다."
```

고 가정하지 않는다.

각 접근마다:

```text
Who?
What device?
What permission?
```

을 검증하는 관점이 Zero Trust와 연결된다.

---

# 100. Robot Internal Network도 무조건 신뢰하지 않는다

Robot 안에 연결된 device도 compromise될 가능성이 있다.

예:

```text
Camera module
Third-party sensor
External laptop
```

따라서 내부 network라고 무조건 모든 traffic을 신뢰하면 안 된다.

---

# 101. Sensor Spoofing

공격자가 sensor data를 조작하면:

```text
Fake GPS
Fake LiDAR packet
Fake command
```

같은 문제가 생길 수 있다.

Security와 state estimation이 연결되는 부분이다.

---

# 102. Command Authentication

High-level control command가 network를 통해 온다면:

```text
Who sent this command?
```

를 확인할 수 있어야 한다.

---

# 103. Replay Attack

공격자가 과거의 정상 command를 저장했다가 다시 보내는 공격:

```text
Replay Attack
```

이다.

예:

```text
"Move forward"
```

command를 나중에 다시 보내는 것이다.

---

# 104. Replay 방어

프로토콜에 따라:

```text
Timestamp
Nonce
Sequence Number
Session Key
```

등을 사용할 수 있다.

---

# 105. Security와 Time Synchronization

Chapter 11의 time sync는 security에서도 중요하다.

Certificate validity나 replay prevention에서
정확한 clock이 필요할 수 있다.

---

# 106. Certificate Time

Certificate에는 validity period가 있을 수 있다.

```text
Not Before
Not After
```

Robot clock이 크게 틀리면 정상 certificate도 invalid로 판단할 수 있다.

---

# 107. Data at Rest

Storage에 저장된 data:

```text
rosbag
Map
Log
Video
Credentials
```

를:

```text
Data at Rest
```

라고 한다.

---

# 108. Data in Transit

Network를 통해 이동하는 data:

```text
ROS message
Cloud telemetry
Video stream
SSH
```

를:

```text
Data in Transit
```

이라고 한다.

---

# 109. 보호 방법

```text
Data at Rest
→ Disk encryption / file permission

Data in Transit
→ TLS / VPN / DDS Security
```

등을 사용할 수 있다.

---

# 110. File Permission

Linux에서는 중요한 config/secret file에
적절한 permission을 설정한다.

예:

```bash
chmod 600 private_key
```

의미:

```text
Owner read/write only
```

---

# 111. `chmod 777` 문제

편하다고:

```bash
chmod 777 ...
```

를 자주 사용하면 누구나 읽고 쓰고 실행할 수 있는 상태가 될 수 있다.

Production에서는 피하는 것이 좋다.

---

# 112. Ownership

확인:

```bash
ls -l
```

또는:

```bash
stat file
```

Owner와 group을 확인한다.

---

# 113. Secrets의 Backup

Secret도 backup이 필요할 수 있지만:

```text
Plain text copy everywhere
```

는 위험하다.

암호화된 backup과 access policy가 필요하다.

---

# 114. Credential Rotation

Password, token, key를 일정 주기로 교체하거나
사고 발생 시 즉시 교체할 수 있어야 한다.

이를:

```text
Credential Rotation
```

이라고 한다.

---

# 115. Employee Offboarding

회사 계정이나 SSH key를 가진 사람이 더 이상 접근할 필요가 없다면:

```text
Key removal
Account disable
Token revoke
```

가 필요하다.

---

# 116. Shared Account 문제

여러 사람이:

```text
robot / password123
```

하나의 account를 공유하면
누가 어떤 작업을 했는지 알기 어렵다.

가능하면 individual account/access를 사용하는 것이 좋다.

---

# 117. Environment Separation

개발용과 production credential을 분리한다.

```text
Development
Staging
Production
```

환경마다 access 권한을 다르게 한다.

---

# 118. Production Robot에 Debug Credential 남기지 않는다

개발 중 사용한:

```text
Test password
Debug SSH key
Open debug port
```

를 production에 남기지 않는 것이 중요하다.

---

# 119. Security Baseline

Robot 출고/배포 전에 최소한 다음을 확인한다.

```text
Default password removed
SSH key configured
Unused user removed
Unused port closed
Firewall policy set
Secrets protected
Software versions recorded
Update path defined
```

---

# 120. Incident Response

보안 문제가 발생했을 때:

```text
Detect
   ↓
Isolate
   ↓
Revoke Credential
   ↓
Collect Logs
   ↓
Patch
   ↓
Redeploy
```

절차가 필요하다.

---

# 121. Robot 격리

Compromise가 의심되면:

```text
Network disconnect
Fleet isolation
Credential revoke
```

를 통해 피해 확산을 막을 수 있다.

---

# 122. Fleet Security

Robot이 여러 대라면 보안은 device 단위가 아니라 fleet 단위가 된다.

```text
Fleet

Robot A
Robot B
Robot C
...
```

필요한 것:

```text
Unique identity
Version tracking
Credential management
Update management
Monitoring
```

---

# 123. AWS IoT 예제

Cloud IoT system에서는 각 robot에:

```text
Thing / Device Identity
Certificate
Private Key
Policy
```

를 줄 수 있다.

구조:

```text
Robot
Certificate
   │
   │ TLS
   ▼
IoT Service
```

---

# 124. Policy

각 device가 어떤 cloud resource를 사용할 수 있는지 제한한다.

예:

```text
Robot A
→ publish /robotA/telemetry
→ subscribe /robotA/commands

Robot B
→ /robotB/*
```

---

# 125. Least Privilege in Cloud

Robot credential에:

```text
All S3
All IoT
All Cloud resources
```

권한을 주는 것보다

```text
필요한 topic
필요한 bucket
필요한 action
```

만 허용하는 것이 좋다.

---

# 126. Security vs Convenience

개발에서는:

```text
root
777
--privileged
firewall off
```

가 편할 수 있다.

하지만 production에서는 위험하다.

따라서:

```text
Development Convenience
vs
Production Security
```

를 구분해야 한다.

---

# 127. Security vs Availability

보안을 너무 강하게 설정하면 정상 동작을 막을 수도 있다.

예:

```text
Firewall
      ↓
ROS 2 blocked
```

그래서 실제 traffic requirement를 이해하고
필요한 것만 허용해야 한다.

---

# 128. Security Testing

배포 전 다음을 확인할 수 있다.

```text
Open ports
Default credentials
SSH access
Firewall
Container privileges
File permissions
Secrets
Update signatures
```

---

# 129. Basic Security Inspection Commands

Listening ports:

```bash
ss -tulpen
```

Users:

```bash
cat /etc/passwd
```

Current user:

```bash
whoami
```

Groups:

```bash
groups
```

SSH login history:

```bash
last
```

Firewall:

```bash
sudo ufw status
```

---

# 130. Mini Practice 1

Jetson에서:

```bash
whoami
```

```bash
groups
```

를 확인한다.

질문:

```text
현재 user는 root인가?
어떤 group에 속하는가?
```

---

# 131. Mini Practice 2

```bash
ss -tuln
```

실행한다.

질문:

```text
어떤 port가 listening 중인가?
왜 열려 있는가?
```

모르는 service가 있으면 확인한다.

---

# 132. Mini Practice 3

SSH key directory:

```bash
ls -la ~/.ssh
```

확인.

다음 파일의 차이를 이해한다.

```text
id_ed25519
id_ed25519.pub
authorized_keys
known_hosts
```

---

# 133. Mini Practice 4

Private key permission 확인:

```bash
ls -l ~/.ssh/id_ed25519
```

일반적으로 private key는 다른 user가 읽을 수 없도록 제한해야 한다.

---

# 134. Mini Practice 5

Repository에서 secret 검색 습관을 만든다.

예:

```bash
rg "password|secret|token|private_key|access_key"
```

단순 keyword search라 false positive/false negative가 있을 수 있지만
초기 점검에는 도움이 된다.

---

# 135. Mini Practice 6

Docker container가 있다면:

```bash
docker inspect <container>
```

확인.

질문:

```text
Privileged인가?
어떤 device를 mount했는가?
Host network를 쓰는가?
Secret environment variable이 있는가?
```

---

# 136. Mini Practice 7

Robot access architecture를 그린다.

예:

```text
Developer Laptop
      │
      ▼
Company Network
      │
      ▼
Jetson Orin
      │
      ▼
Robot Internal Network
      │
      ├── Xavier
      ├── LiDAR
      └── Other Devices
```

각 연결에 대해:

```text
Who can access?
What protocol?
What authentication?
```

을 작성한다.

---

# 137. 반드시 구분할 것

```text
Authentication
≠
Authorization

Public Key
≠
Private Key

Certificate
≠
Private Key

ROS_DOMAIN_ID
≠
Security

Firewall
≠
Encryption

Hash
≠
Digital Signature

Secure Boot
≠
Disk Encryption

Container
≠
Perfect Security Boundary

Internal Network
≠
Automatically Trusted

Development Access
≠
Production Access
```

---

# 138. Security Mental Model

Robot security를 layer별로 본다.

```text
Physical Security
       │
       ▼
Secure Boot
       │
       ▼
Linux Users / Permissions
       │
       ▼
Container Isolation
       │
       ▼
Network Firewall
       │
       ▼
Authentication
       │
       ▼
Encryption
       │
       ▼
Application Authorization
       │
       ▼
Cloud Policy
```

---

# 139. Vision60 Example

개념적인 구조:

```text
                   Developer
                       │
                  SSH Key / VPN
                       │
                       ▼
                 Company Network
                       │
                       ▼
                 ┌──────────┐
                 │   Orin   │
                 │          │
                 │ Firewall │
                 │ SSH      │
                 └────┬─────┘
                      │
               Robot Internal LAN
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Xavier       LiDAR       Other
          │
          ▼
       ROS 2
          │
          ▼
     Robot Control
```

가능하면 외부 user가:

```text
LiDAR
MCU
Low-Level Controller
```

에 직접 접근할 필요가 없도록 설계한다.

---

# 140. Security의 가장 중요한 질문

각 system component에 대해 항상 묻는다.

```text
Who can access it?

How are they authenticated?

What are they allowed to do?

Is traffic encrypted?

Where are secrets stored?

What happens if the device is stolen?

How do we revoke access?

How do we update securely?
```

---

# 141. Chapter 16 핵심

Embedded Security는 단순히 password를 설정하는 문제가 아니다.

전체 lifecycle을 봐야 한다.

```text
Manufacture / Provision
        ↓
Deploy
        ↓
Authenticate
        ↓
Operate
        ↓
Monitor
        ↓
Update
        ↓
Revoke
        ↓
Retire
```

Robot이 network와 cloud에 연결될수록
보안도 software architecture의 일부가 된다.

---

# Next Chapter

## Chapter 17. Remote Deployment & Fleet Management

다음 Chapter에서는 robot이 여러 대가 되었을 때
software를 어떻게 운영하는지 다룬다.

```text
Fleet
Device Identity
Provisioning
OTA Update
Container Registry
Versioning
Staged Rollout
Rollback
Telemetry
Remote Command
AWS IoT
Greengrass
Device Group
```

특히:

```text
Developer
    ↓
CI/CD
    ↓
Container Registry
    ↓
Robot Fleet
```

구조를 배우고,

```text
"Robot 100대를 어떻게 같은 version으로 유지하는가?"

"Update가 실패하면 어떻게 이전 version으로 돌아가는가?"

"각 robot의 상태와 log를 어떻게 원격으로 확인하는가?"
```

를 다룬다.