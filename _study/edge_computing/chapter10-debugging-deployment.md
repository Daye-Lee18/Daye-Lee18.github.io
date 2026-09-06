---
title: "Chapter 10. ROS 2 + Jetson Debugging & Deployment"
importance: 11
---
> **Goal:** 로봇에서 문제가 생겼을 때 감으로 디버깅하지 않고
> Hardware → Linux → Network → Docker → ROS 2 → CUDA → Application 순서로
> 문제를 체계적으로 좁히는 방법을 익힌다.
>
> 또한 개발 환경에서 만든 software를 Jetson에 안정적으로 배포하고 운영하는 기본 전략을 이해한다.

---

# 1. 로봇 문제는 한 Layer에서만 생기지 않는다

예를 들어 FAST-LIO2가 제대로 동작하지 않는다고 하자.

원인은:

```text
FAST-LIO2 code
```

일 수도 있지만,

```text
LiDAR cable
Network
ROS 2 QoS
Docker network
CPU overload
Thermal throttling
```

문제일 수도 있다.

따라서 로봇 debugging에서는 전체 stack을 봐야 한다.

```text
Application
   ↓
ROS 2
   ↓
Docker
   ↓
Network
   ↓
Linux
   ↓
Hardware
```

---

# 2. 가장 중요한 원칙

문제가 생겼을 때 바로 application code부터 고치지 않는다.

먼저 아래 layer부터 위로 올라간다.

```text
1. Hardware
2. Linux
3. Network
4. Device / Driver
5. Docker
6. ROS 2
7. Application
8. Performance
```

---

# 3. Hardware부터 확인

예를 들어 LiDAR가 안 들어온다고 하자.

먼저:

```text
Power 들어왔는가?
Cable 연결됐는가?
Ethernet link가 살아 있는가?
Sensor LED는 정상인가?
```

를 확인한다.

Software가 아무리 정상이어도 hardware가 연결되지 않았다면 동작하지 않는다.

---

# 4. Linux가 Device를 보는가?

USB sensor라면:

```bash
lsusb
```

Serial:

```bash
ls /dev/ttyUSB*
```

Camera:

```bash
ls /dev/video*
```

PCIe device:

```bash
lspci
```

Storage:

```bash
lsblk
```

등을 확인한다.

---

# 5. Network Device라면

LiDAR가 Ethernet sensor라면:

```bash
ip addr
```

```bash
ip route
```

```bash
ping <lidar-ip>
```

를 확인한다.

예:

```bash
ping 192.168.10.20
```

응답이 없다면 ROS 2를 보기 전에 network 문제부터 해결해야 한다.

---

# 6. Ethernet Link 확인

```bash
ethtool eth0
```

예:

```text
Speed: 1000Mb/s
Duplex: Full
Link detected: yes
```

확인한다.

```text
Link detected: no
```

라면 cable, switch, port, sensor power 문제일 수 있다.

---

# 7. IP가 있어도 Route가 틀릴 수 있다

예를 들어 Jetson:

```text
192.168.10.10/24
```

LiDAR:

```text
192.168.10.20/24
```

이면 같은 subnet이다.

하지만 route가 이상하면 통신이 안 될 수 있다.

확인:

```bash
ip route
```

---

# 8. Driver가 실행 중인가?

Network가 정상이라면 driver를 확인한다.

예:

```bash
ps aux | grep lidar
```

또는 ROS 2:

```bash
ros2 node list
```

로 driver node가 있는지 본다.

---

# 9. ROS 2 Topic 존재 확인

```bash
ros2 topic list
```

LiDAR topic이 존재하는지 확인한다.

예:

```text
/velodyne_points
```

---

# 10. Topic이 있다고 Data가 있다는 뜻은 아니다

확인:

```bash
ros2 topic hz /velodyne_points
```

또는:

```bash
ros2 topic echo /velodyne_points
```

Point cloud는 너무 크므로 `echo` 대신 `hz`나 `info`가 더 실용적일 수 있다.

---

# 11. Topic Type 확인

```bash
ros2 topic type /velodyne_points
```

예:

```text
sensor_msgs/msg/PointCloud2
```

FAST-LIO2가 기대하는 message type과 맞는지 확인한다.

---

# 12. Publisher와 Subscriber 확인

```bash
ros2 topic info /velodyne_points --verbose
```

확인할 것:

```text
Publisher count
Subscriber count
QoS
Node name
```

FAST-LIO2가 subscriber로 연결되어 있는지 확인한다.

---

# 13. QoS 확인

ROS 2에서는 topic 이름이 같아도 QoS가 맞지 않으면 communication이 안 될 수 있다.

특히 sensor data:

```text
LiDAR
IMU
Camera
```

에서 중요하다.

예:

```text
Publisher:
Best Effort

Subscriber:
Reliable only
```

이면 compatibility 문제가 생길 수 있다.

---

# 14. ROS_DOMAIN_ID 확인

Multi-computer 환경에서는:

```bash
echo $ROS_DOMAIN_ID
```

를 확인한다.

Xavier:

```text
123
```

Orin:

```text
123
```

처럼 맞아야 하는 경우가 많다.

---

# 15. RMW 확인

```bash
echo $RMW_IMPLEMENTATION
```

예:

```text
rmw_cyclonedds_cpp
```

ROS 2 discovery 문제가 있다면 RMW와 DDS configuration을 같이 본다.

---

# 16. CycloneDDS 설정

환경에 따라:

```bash
echo $CYCLONEDDS_URI
```

를 확인한다.

예:

```text
/home/robot/vision60_ws/cyclonedds.xml
```

DDS가 잘못된 network interface를 선택하면 node discovery 문제가 생길 수 있다.

---

# 17. 여러 Network Interface 문제

Jetson:

```text
eth0
wlan0
docker0
```

가 동시에 존재할 수 있다.

DDS가:

```text
docker0
```

같은 잘못된 interface를 잡으면 다른 computer와 communication이 안 될 수 있다.

따라서 interface 선택을 확인해야 한다.

---

# 18. Docker를 사용한다면

Host에서는 ROS topic이 보이는데 container 안에서는 안 보일 수 있다.

확인:

```bash
docker ps
```

그리고 container 내부:

```bash
docker exec -it <container> bash
```

에서:

```bash
ros2 topic list
```

확인한다.

---

# 19. Docker Network 확인

```bash
docker inspect <container>
```

또는:

```bash
docker network ls
```

확인한다.

ROS 2에서는:

```text
bridge network
vs
host network
```

차이가 매우 중요할 수 있다.

---

# 20. `--network host`

ROS 2 multi-machine communication이 필요하다면:

```bash
--network host
```

를 사용하는 구성이 간단할 수 있다.

하지만 security와 port conflict를 고려해야 한다.

---

# 21. Device가 Container에 들어왔는가?

예:

```text
/dev/video0
/dev/ttyUSB0
```

를 사용하는 node라면 container 안에서:

```bash
ls /dev/video*
```

등을 확인한다.

Host에서는 보이는데 container에서는 안 보이면
Docker device mapping 문제다.

---

# 22. Permission 문제

Device가 존재해도:

```text
Permission denied
```

가 날 수 있다.

확인:

```bash
ls -l /dev/ttyUSB0
```

예:

```text
crw-rw---- root dialout ...
```

사용자가 해당 group에 속하는지 확인할 수 있다.

---

# 23. `sudo`로만 해결하지 않는다

무조건:

```bash
sudo ...
```

를 붙이면 일시적으로 해결될 수 있지만
근본적인 permission 문제가 가려질 수 있다.

Production에서는 proper user/group permission을 사용하는 것이 좋다.

---

# 24. Process 확인

```bash
ps aux
```

또는:

```bash
htop
```

으로 필요한 process가 실제로 실행 중인지 확인한다.

예:

```text
velodyne_driver
fastlio_mapping
nav2
```

---

# 25. Duplicate Process

로봇에서는 같은 driver나 node가 중복 실행되면 문제가 생길 수 있다.

예:

```text
velodyne_transform A
velodyne_transform B
```

둘이 같은 resource나 topic을 다루면 이상한 결과가 발생할 수 있다.

확인:

```bash
ps aux | grep velodyne
```

```bash
ros2 node list
```

---

# 26. CPU 사용량 확인

```bash
htop
```

또는:

```bash
top
```

를 사용한다.

FAST-LIO2가:

```text
100% CPU
```

를 사용하고 있다면 compute bottleneck일 수 있다.

멀티코어 시스템에서 100% 의미는 tool마다 다를 수 있으므로
전체 core 사용량과 per-core usage를 같이 본다.

---

# 27. Memory 확인

```bash
free -h
```

예:

```text
Mem:
32Gi total
30Gi used
```

RAM이 거의 꽉 차면:

```text
Swap
OOM
Performance degradation
```

문제가 발생할 수 있다.

---

# 28. OOM

OOM은:

**Out Of Memory**

이다.

Linux가 memory를 충분히 확보하지 못하면 process를 강제로 종료할 수도 있다.

확인:

```bash
dmesg
```

또는 system log에서 OOM killer 관련 메시지를 찾는다.

---

# 29. Disk 확인

```bash
df -h
```

rosbag이나 log가 쌓여 disk가 가득 찰 수 있다.

예:

```text
/dev/nvme0n1p1  100%
```

이면 새로운 log나 map을 저장하지 못할 수 있다.

---

# 30. Directory별 사용량

```bash
du -sh *
```

또는:

```bash
du -sh /path/to/logs
```

로 어떤 directory가 많은 storage를 사용하는지 확인한다.

---

# 31. Log 확인

System service:

```bash
journalctl
```

특정 service:

```bash
journalctl -u <service>
```

실시간:

```bash
journalctl -u <service> -f
```

Docker:

```bash
docker logs <container>
```

ROS 2는 stdout/log file을 확인한다.

---

# 32. `dmesg`

Kernel message 확인:

```bash
dmesg
```

Hardware, USB, driver, network link 문제를 찾는 데 유용하다.

예:

```text
USB device disconnected
network link down
out of memory
```

---

# 33. Jetson Resource 확인

Jetson에서는:

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
Memory controller
Power
```

---

# 34. Thermal 문제

프로그램이 처음에는 빠르다가 시간이 지나면 느려진다면
thermal throttling 가능성이 있다.

구조:

```text
High Load
   ↓
Temperature ↑
   ↓
Clock ↓
   ↓
Performance ↓
```

---

# 35. Power Mode 확인

```bash
sudo nvpmodel -q
```

현재 Jetson power mode를 확인한다.

Benchmark 결과를 비교할 때 power mode가 다르면
성능 차이가 크게 날 수 있다.

---

# 36. GPU가 실제로 사용되는가?

PyTorch:

```python
import torch
print(torch.cuda.is_available())
```

그리고:

```bash
tegrastats
```

로 GPU activity를 본다.

GPU가 있다고 application이 자동으로 GPU를 사용하는 것은 아니다.

---

# 37. CUDA 문제

확인:

```bash
nvcc --version
```

하지만 `nvcc`가 있다고 PyTorch가 반드시 CUDA를 사용할 수 있는 것은 아니다.

확인해야 할 것:

```text
JetPack
CUDA
Driver
PyTorch build
Container image
Architecture
```

이다.

---

# 38. Application Layer로 올라가기

Hardware부터 ROS 2까지 정상이라면
이제 application 내부를 본다.

FAST-LIO2 예:

```text
LiDAR callback called?
IMU callback called?
sync_packages succeeds?
undistortion works?
filter update runs?
odometry publishes?
```

---

# 39. Callback 확인

FAST-LIO2 코드에서:

```text
imu_cbk
standard_pcl_cbk
sync_packages
UndistortPcl
publish_odometry
```

등을 순서대로 본다.

Mental model:

```text
LiDAR arrival
   ↓
LiDAR callback
   ↓
Buffer
   ↓
IMU synchronization
   ↓
Deskew
   ↓
State estimation
   ↓
Map update
   ↓
Odometry publish
```

---

# 40. Input과 Output을 분리해서 본다

Application debugging에서는:

```text
Input 정상?
Processing 정상?
Output 정상?
```

세 단계로 나누면 좋다.

예:

```text
FAST-LIO2

Input
LiDAR / IMU
   ↓
Processing
ESKF / Mapping
   ↓
Output
Odometry / Point Cloud
```

---

# 41. Timestamp 문제

LiDAR와 IMU fusion에서는 timestamp가 중요하다.

예:

```text
LiDAR time: 100.0
IMU time:   95.0
```

처럼 크게 어긋나면 synchronization이 실패할 수 있다.

확인할 것:

```text
Sensor clock
System clock
Message header stamp
Time synchronization
```

---

# 42. Multi-computer Clock

Xavier와 Orin에서 센서 처리를 나눠 한다면
computer clock도 중요할 수 있다.

예:

```text
Xavier clock
10:00:00.000

Orin clock
10:00:03.000
```

3초 차이라면 sensor fusion에 큰 문제가 생길 수 있다.

NTP/PTP 같은 time synchronization 기술을 사용할 수 있다.

---

# 43. NTP

NTP:

**Network Time Protocol**

network를 통해 computer clock을 맞추는 protocol이다.

일반적인 system time synchronization에 많이 사용된다.

---

# 44. PTP

PTP:

**Precision Time Protocol**

더 높은 정밀도의 time synchronization이 필요한 industrial/robotics 환경에서 사용할 수 있다.

Hardware timestamping과 함께 사용할 수도 있다.

---

# 45. Frame 문제

ROS 2 SLAM에서는:

```text
map
odom
base_link
lidar_link
imu_link
```

frame이 중요하다.

확인:

```bash
ros2 run tf2_tools view_frames
```

환경에 따라 available package가 필요하다.

---

# 46. TF 문제

예:

```text
base_link → lidar_link
```

transform이 잘못되어 있으면 map이 이상하게 생성될 수 있다.

Sensor 자체는 정상이어도 calibration/TF가 틀릴 수 있다.

---

# 47. Calibration

Sensor 간 position/orientation 관계를 정확히 알아야 한다.

예:

```text
IMU ↔ LiDAR
```

extrinsic calibration이 틀리면 SLAM accuracy가 떨어진다.

즉:

```text
Data arrives
≠
Data is geometrically correct
```

이다.

---

# 48. Parameter 문제

ROS parameter:

```bash
ros2 param list
```

확인.

FAST-LIO2 parameter:

```text
LiDAR type
scan line
extrinsic
noise covariance
filter setting
```

등이 잘못되면 software는 실행되지만 결과가 나쁠 수 있다.

---

# 49. Config File 확인

실제로 어떤 YAML이 로드되는지 확인해야 한다.

예:

```text
config/default.yaml
config/vision60.yaml
config/site.yaml
```

여러 config가 있으면 잘못된 파일이 로드되는 경우도 많다.

---

# 50. Environment Variable 문제

확인:

```bash
env | sort
```

ROS:

```bash
echo $ROS_DOMAIN_ID
```

```bash
echo $RMW_IMPLEMENTATION
```

Workspace:

```bash
echo $AMENT_PREFIX_PATH
```

잘못된 environment가 package나 middleware 선택에 영향을 줄 수 있다.

---

# 51. Source 순서

예:

```bash
source /opt/ros/humble/setup.bash
source ~/vision60_ws/install/setup.bash
```

순서가 중요하다.

Overlay workspace가 system ROS 위에 올라가도록 하는 경우가 일반적이다.

---

# 52. 어떤 Executable이 실행되는가?

```bash
which python3
```

```bash
which ros2
```

를 확인한다.

동일한 이름의 executable이 여러 곳에 있으면
생각과 다른 프로그램이 실행될 수 있다.

---

# 53. Library Dependency

C++ binary:

```bash
ldd <binary>
```

를 사용해 dynamic library dependency를 확인할 수 있다.

예:

```text
libfoo.so => not found
```

이면 library path 문제다.

---

# 54. Python Environment

Python:

```bash
which python
```

```bash
python --version
```

```bash
pip show <package>
```

확인.

System Python, Conda, venv, container가 섞이면 dependency confusion이 생길 수 있다.

---

# 55. Reproducibility

개발자의 laptop에서만 돌아가는 software는 좋은 deployment 상태가 아니다.

목표:

```text
Same Source
Same Dependency
Same Config
Same Result
```

를 최대한 만들기.

Docker가 여기에 도움이 된다.

---

# 56. Development와 Deployment를 구분

Development:

```text
Source mount
Debug tool
Interactive shell
Changing code
```

Deployment:

```text
Fixed image
Fixed config
Automatic startup
Logging
Health monitoring
Rollback
```

---

# 57. Git Commit 기준 배포

Production robot에는:

```text
latest source
```

보다 명확한 commit을 사용하는 것이 좋다.

예:

```text
commit:
a1b2c3d
```

이렇게 하면 어느 code가 배포되었는지 추적할 수 있다.

---

# 58. Versioning

예:

```text
vision60-autonomy:v1.2.0
```

같은 version을 사용할 수 있다.

배포 상태를:

```text
Robot A → v1.2.0
Robot B → v1.1.3
```

처럼 추적할 수 있다.

---

# 59. Configuration Version

Code version만 관리하면 부족하다.

예:

```text
Software v1.2.0
Config v3
Map version 2026-09-05
```

같이 관리할 수 있다.

---

# 60. Immutable Deployment

Production에서는 실행 중 robot에서 package를 직접 수정하기보다
새 image/version을 배포하는 것이 재현성과 rollback 측면에서 좋다.

```text
Old Image
   ↓
New Image
```

---

# 61. Rollback

새 버전이 문제가 생기면 이전 버전으로 돌아갈 수 있어야 한다.

예:

```text
v1.2.0
  ↓ failed
v1.1.3
```

이것이 production deployment에서 중요하다.

---

# 62. Startup

Robot이 reboot되면 필요한 software가 자동으로 시작되어야 할 수 있다.

방법:

```text
systemd
Docker restart policy
Supervisor
```

등.

---

# 63. systemd 기반 실행

예:

```text
Boot
 ↓
systemd
 ↓
vision60.service
 ↓
Docker / ROS 2
```

장점:

```text
Auto-start
Restart
Logging
Dependency ordering
```

---

# 64. Startup Ordering

예를 들어:

```text
Network
   ↓
LiDAR
   ↓
ROS 2 Driver
   ↓
FAST-LIO2
   ↓
Navigation
```

순서가 필요할 수 있다.

단순 sleep보다 readiness check를 사용하는 것이 더 좋다.

---

# 65. `sleep 10`의 문제

예:

```bash
sleep 10
ros2 launch ...
```

은 간단하지만 network가 15초 걸리면 실패한다.

더 좋은 방식:

```text
Wait until condition ready
```

예:

```text
LiDAR reachable?
Topic exists?
Service ready?
```

를 확인한다.

---

# 66. Health Check

프로그램이 process로 살아 있다는 것만 확인하면 부족하다.

예:

```text
FAST-LIO2 process alive
```

여도:

```text
LiDAR topic 0 Hz
```

일 수 있다.

따라서 health check는 application 의미까지 보는 것이 좋다.

---

# 67. Health Check 예

```text
LiDAR topic > 5 Hz?
IMU topic > 100 Hz?
Odometry updating?
Disk < 90%?
Temperature normal?
```

등.

---

# 68. Watchdog

특정 component가 멈췄을 때 자동으로 감지하고 재시작하는 구조를 만들 수 있다.

```text
Application
   │
Heartbeat
   ▼
Watchdog
   │
No heartbeat
   ▼
Restart
```

---

# 69. 자동 재시작도 무조건 좋은 것은 아니다

무한 restart가 발생하면 원인을 가릴 수 있다.

예:

```text
Crash
Restart
Crash
Restart
...
```

따라서:

```text
Restart count
Backoff
Alert
Logs
```

가 필요하다.

---

# 70. Logging

Production robot에서는 debugging을 위해 log가 매우 중요하다.

좋은 log에는:

```text
Timestamp
Severity
Node
Error reason
Relevant state
```

가 포함되어야 한다.

---

# 71. Log Level

일반적으로:

```text
DEBUG
INFO
WARN
ERROR
FATAL
```

같은 level을 사용한다.

Production에서는 DEBUG를 항상 켜두면 log가 너무 커질 수 있다.

---

# 72. Log Rotation

Robot이 장시간 동작하면 log가 disk를 가득 채울 수 있다.

따라서:

```text
Maximum file size
Retention period
Number of files
```

을 제한한다.

---

# 73. rosbag

문제가 재현되기 어렵다면 sensor data를 rosbag으로 기록할 수 있다.

```bash
ros2 bag record ...
```

나중에:

```text
Same sensor input
   ↓
Offline replay
   ↓
Algorithm debugging
```

이 가능하다.

---

# 74. Record Everything은 위험

모든 topic을 장시간 기록하면 storage가 매우 빨리 찬다.

특히:

```text
Camera
Point Cloud
```

는 용량이 크다.

필요한 topic과 duration을 정하는 것이 좋다.

---

# 75. Failure Reproduction

좋은 debugging에서는:

```text
"가끔 안 돼요"
```

를:

```text
이 sensor input에서
이 config로
이 commit을 실행하면
37초 후 failure
```

로 바꿔야 한다.

---

# 76. Baseline 만들기

새로운 algorithm을 테스트할 때 정상 상태를 먼저 기록한다.

예:

```text
FAST-LIO2 baseline

CPU: 120%
RAM: 4.2 GB
LiDAR: 10 Hz
IMU: 200 Hz
Odometry: 100 Hz
Temperature: 58 C
```

이후 변화와 비교한다.

---

# 77. Performance Regression

새 코드 이후:

```text
CPU: 120% → 250%
Odometry: 100 Hz → 40 Hz
```

가 되었다면 regression이 생긴 것이다.

따라서 기능뿐 아니라 performance도 테스트해야 한다.

---

# 78. Benchmark 조건 고정

비교할 때:

```text
Same Jetson
Same Power Mode
Same Dataset
Same Config
Same Software Version
Same Temperature condition
```

을 맞춘다.

안 그러면 공정한 비교가 아니다.

---

# 79. Mean만 보면 안 된다

Latency:

```text
Average: 10 ms
```

만 보는 것보다:

```text
Median
P95
P99
Max
```

같은 분포를 보는 것이 중요하다.

로봇에서는 occasional latency spike가 위험할 수 있다.

---

# 80. Jitter 측정

예:

```text
9 ms
10 ms
9 ms
11 ms
80 ms
```

평균은 괜찮아 보여도 80 ms spike가 문제일 수 있다.

---

# 81. ROS 2 Frequency Monitoring

예:

```bash
ros2 topic hz /imu
```

Expected:

```text
200 Hz
```

Actual:

```text
80 Hz
```

라면 bottleneck이나 packet loss를 의심할 수 있다.

---

# 82. Network Bandwidth

Camera/LiDAR가 많다면 network bandwidth도 모니터링한다.

도구 예:

```bash
iftop
```

```bash
nload
```

설치되어 있다면 사용할 수 있다.

---

# 83. CPU Affinity

특정 process를 특정 CPU에 배치하는 최적화도 가능하다.

예:

```bash
taskset
```

하지만 일반적인 첫 번째 해결책은 아니다.

먼저 profiling 후 필요할 때 적용한다.

---

# 84. Process Priority

Linux scheduling priority를 조정할 수도 있다.

예:

```text
nice
realtime scheduling
```

하지만 real-time priority는 system 전체에 영향을 줄 수 있어
충분히 이해한 후 사용해야 한다.

---

# 85. Real-Time 문제

Hard real-time motor control과
Jetson의 general Linux application은 요구사항이 다르다.

```text
Jetson
→ Perception / Planning / SLAM

MCU / RT Controller
→ Motor control
```

역할 분리가 중요한 이유다.

---

# 86. Deployment Checklist

Robot에 새 software를 배포하기 전:

```text
Code commit fixed?
Config fixed?
Image tag fixed?
Architecture correct?
JetPack compatible?
ROS version correct?
Network config correct?
Disk enough?
Rollback possible?
```

확인한다.

---

# 87. Preflight Check

로봇 시작 전에 자동 확인할 수 있다.

예:

```text
LiDAR reachable
IMU available
Disk OK
Temperature OK
ROS_DOMAIN_ID correct
Map exists
```

하나라도 실패하면 autonomy stack 시작을 막을 수도 있다.

---

# 88. Safe Failure

Robot software는 문제가 생겼을 때
가능하면 안전한 상태로 가야 한다.

예:

```text
Localization lost
      ↓
Stop autonomy
      ↓
Robot stop / operator intervention
```

잘못된 pose로 계속 이동하는 것보다 안전하다.

---

# 89. Observability

System 내부 상태를 외부에서 확인할 수 있어야 한다.

예:

```text
Logs
Metrics
Health
Topic rates
CPU
GPU
Temperature
Network
```

이것을 observability라고 한다.

---

# 90. Remote Debugging

회사 network에서 robot에 접근할 수 있으면:

```text
SSH
Logs
Metrics
Deployment
```

가 편해진다.

하지만 network exposure가 늘어나는 만큼 security를 반드시 고려해야 한다.

---

# 91. Production Security

최소한 다음은 확인한다.

```text
Default password 제거
SSH key 사용
불필요 port 닫기
Secrets repository에 넣지 않기
Software update 관리
Least privilege
```

---

# 92. Secret 관리

다음은 Git에 올리면 안 된다.

```text
SSH private key
Cloud access key
Password
API token
Certificate private key
```

특히 public repository에는 절대 넣지 않는다.

---

# 93. Robot Debugging Flow

전체 flow:

```text
Robot not working
      │
      ▼
Hardware OK?
      │
      ▼
Linux sees device?
      │
      ▼
Network reachable?
      │
      ▼
Driver alive?
      │
      ▼
Docker configured?
      │
      ▼
ROS node/topic?
      │
      ▼
QoS / DDS?
      │
      ▼
Application input?
      │
      ▼
Application processing?
      │
      ▼
Output?
      │
      ▼
Performance?
```

---

# 94. FAST-LIO2 Debugging Flow

FAST-LIO2가 안 될 때:

```text
LiDAR powered?
      ↓
LiDAR network reachable?
      ↓
LiDAR driver running?
      ↓
PointCloud topic Hz normal?
      ↓
IMU topic Hz normal?
      ↓
Timestamp sane?
      ↓
FAST-LIO2 subscribers connected?
      ↓
Callbacks running?
      ↓
Synchronization works?
      ↓
Extrinsic correct?
      ↓
Filter update?
      ↓
Odometry publish?
      ↓
TF correct?
```

---

# 95. Navigation Debugging Flow

Robot이 목표 지점으로 안 간다면:

```text
Localization valid?
      ↓
map/odom/base_link TF valid?
      ↓
Costmap valid?
      ↓
Planner produces path?
      ↓
Controller produces cmd?
      ↓
Robot receives cmd?
      ↓
Low-level controller works?
```

단순히 Nav2 bug라고 생각하지 않는다.

---

# 96. Camera AI Debugging Flow

```text
Camera powered?
   ↓
Linux /dev/video?
   ↓
Driver frames?
   ↓
ROS Image topic?
   ↓
Container device access?
   ↓
CUDA available?
   ↓
Model loaded?
   ↓
Inference latency?
   ↓
Output published?
```

---

# 97. Debugging 기록

문제를 해결할 때 다음을 기록하면 좋다.

```text
Date
Robot
Software commit
JetPack version
Configuration
Symptom
Commands executed
Root cause
Fix
```

나중에 같은 문제가 반복되면 빠르게 해결할 수 있다.

---

# 98. 좋은 Bug Report 예

나쁜 예:

```text
SLAM 안 돼요.
```

좋은 예:

```text
Jetson AGX Orin
JetPack X
ROS 2 Humble
commit abc123

/imu = 200 Hz
/points = 10 Hz

FAST-LIO2 process alive
but /Odometry = 0 Hz

imu_cbk runs,
sync_packages returns false continuously.
```

이렇게 쓰면 문제를 훨씬 빨리 좁힐 수 있다.

---

# 99. 전체 Deployment Architecture

예:

```text
                 Developer

Git Repository
      │
      ▼
CI Build
      │
      ▼
ARM64 Docker Image
      │
      ▼
Container Registry
      │
      ▼
                    Vision60
                       │
                       ▼
                 Jetson Orin
                       │
                       ▼
                    Docker
                       │
                       ▼
                 ROS 2 Stack
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         Driver     FAST-LIO2    Nav2
```

---

# 100. 전체 학습 Stack

지금까지 Chapter 1~10을 하나로 묶으면:

```text
Chapter 1
CPU / GPU / RAM / Storage

        ↓

Chapter 2
ARM64 / x86

        ↓

Chapter 3
Linux

        ↓

Chapter 4
Jetson / JetPack

        ↓

Chapter 5
Ethernet / CAN / USB / PCIe

        ↓

Chapter 6
ROS 2 / RMW / DDS

        ↓

Chapter 7
CUDA / TensorRT

        ↓

Chapter 8
Robot Networking

        ↓

Chapter 9
Docker on Jetson

        ↓

Chapter 10
Debugging & Deployment
```

이제 각각이 따로 떨어진 개념이 아니라
하나의 robot computing stack으로 연결된다.

---

# 101. Final Mental Model

Vision60 전체를 보면:

```text
                           Vision60

Sensors
│
├── LiDAR
├── IMU
├── Camera
└── Joint Sensors
        │
        ▼
Hardware Interfaces
Ethernet / USB / CAN
        │
        ▼
Linux / Driver
        │
        ▼
Docker
        │
        ▼
ROS 2
Node / Topic / DDS
        │
        ▼
Applications
FAST-LIO2 / AI / Nav2
        │
        ▼
State / Command
        │
        ▼
MCU / Controller
        │
        ▼
Actuators
```

그리고 옆에서는 계속:

```text
CPU
GPU
RAM
Network
Storage
Temperature
Power
```

를 monitoring한다.

---

# 102. 가장 중요한 Debugging 질문

앞으로 robot에서 문제가 생기면 먼저 묻는다.

```text
Which layer is broken?
```

그리고:

```text
Hardware?
Linux?
Network?
Docker?
ROS 2?
Application?
Performance?
```

를 하나씩 제거해 나간다.

이 방식이 복잡한 robot system을 가장 안정적으로 debugging하는 기본 사고방식이다.

---

# Course Summary

이 10개 Chapter를 이해했다면
Jetson 기반 로봇 computing stack의 기본 구조는 잡힌 것이다.

다음 단계는 이론을 더 추가하는 것보다
실제 Vision60에 적용하는 것이다.

추천 실습:

```text
Practical 1
Vision60 Network Diagram 직접 작성

Practical 2
Xavier / Orin Hardware Inventory

Practical 3
ROS 2 Graph 전체 추출

Practical 4
FAST-LIO2 Data Flow 추적

Practical 5
CPU / GPU / RAM Profiling

Practical 6
LiDAR / IMU Frequency 측정

Practical 7
Docker Runtime 분석

Practical 8
Robot Boot Sequence 분석

Practical 9
Failure Scenario별 Debugging

Practical 10
Vision60 Full Software Architecture 작성
```