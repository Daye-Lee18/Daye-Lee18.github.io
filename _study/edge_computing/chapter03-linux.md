---
title: "Chapter 3. Linux for Edge Computers"
importance: 4
---
> **Goal:** Jetson에 SSH로 접속했을 때 보이는 Linux 환경을 이해한다.
> 파일 시스템, process, service, permission, environment variable을
> ROS 2와 로봇 개발에 연결해서 이해하는 것이 목표다.

---

# 1. Jetson을 켜면 무엇이 실행될까?

Jetson은 단순한 GPU 보드가 아니다.

Jetson에는 Ubuntu Linux가 설치되어 있고, 우리가 작성한 프로그램은 그 위에서 실행된다.

```text
Robot Hardware
      │
      ▼
┌─────────────────────┐
│       Jetson        │
│                     │
│      Hardware       │
│         ↑           │
│   Linux Kernel      │
│         ↑           │
│      Ubuntu         │
│         ↑           │
│ ROS 2 / FAST-LIO2   │
│ Robot Application   │
└─────────────────────┘
```

예를 들어 Vision60의 Jetson에서 FAST-LIO2를 실행한다면:

```text
FAST-LIO2
    ↓
ROS 2
    ↓
Ubuntu / Linux
    ↓
Jetson Hardware
```

구조라고 생각할 수 있다.

---

# 2. Linux와 Ubuntu는 같은 것인가?

정확히는 다르다.

**Linux**는 핵심적으로 Kernel을 의미하고,
**Ubuntu**는 Linux Kernel을 기반으로 만들어진 Linux Distribution이다.

```text
Ubuntu
├── Linux Kernel
├── GNU utilities
├── package manager
├── system services
├── desktop / shell
└── Ubuntu-specific packages
```

그래서:

```text
Linux
  ↓
Ubuntu
```

라고 단순히 같은 것으로 보기보다는:

> Ubuntu는 Linux를 기반으로 만들어진 운영체제 배포판 중 하나다.

라고 이해하는 것이 좋다.

다른 Linux distribution에는:

```text
Debian
Fedora
Arch Linux
Rocky Linux
```

등이 있다.

Jetson에서는 NVIDIA가 제공하는 JetPack 환경이 Ubuntu 기반으로 구성된다.

---

# 3. Kernel이란?

Kernel은 Operating System의 핵심 부분이다.

Application이 hardware를 직접 마음대로 제어하지 않고,
Kernel을 통해 hardware resource를 사용하도록 관리한다.

```text
Application
ROS 2
FAST-LIO2
     │
     ▼
┌──────────────┐
│ Linux Kernel │
├──────────────┤
│ CPU          │
│ RAM          │
│ Network      │
│ USB          │
│ Storage      │
│ Devices      │
└──────────────┘
```

Kernel의 중요한 역할에는:

- Process scheduling
- Memory management
- Device management
- Filesystem
- Networking
- Permission / security

등이 있다.

Chapter 1에서 CPU core에 여러 프로그램이 실행될 수 있다고 했다.

어떤 process를 어느 CPU core에서 언제 실행할지도 Linux Kernel의 scheduler가 관리한다.

---

# 4. Shell이란?

Jetson에 SSH로 접속하면 보통 이런 화면을 본다.

```bash
daye@jetson:~$
```

여기에:

```bash
ls
cd
mkdir
ps
```

같은 명령어를 입력한다.

이 명령을 받아 해석하는 프로그램을 **Shell**이라고 한다.

대표적인 shell:

```text
bash
zsh
sh
fish
```

Ubuntu에서는 Bash를 매우 자주 사용한다.

확인:

```bash
echo $SHELL
```

예:

```text
/bin/bash
```

---

# 5. Terminal과 Shell은 다르다

둘은 자주 같은 의미처럼 사용되지만 엄밀히는 다르다.

```text
Terminal
   │
   ▼
Shell
   │
   ▼
Command
   │
   ▼
Linux
```

**Terminal**은 문자 기반으로 사용자와 프로그램이 상호작용하는 인터페이스이고,

**Shell**은 입력된 command를 해석하고 실행하는 프로그램이다.

Mac Terminal 안에서 `zsh`를 실행할 수도 있고,
Ubuntu Terminal 안에서 `bash`를 실행할 수도 있다.

---

# 6. Linux Filesystem

Windows에서는:

```text
C:\
D:\
```

같은 drive 개념을 자주 본다.

Linux에서는 모든 filesystem이 하나의 root에서 시작한다.

```text
/
```

이 `/`를 **Root Directory**라고 한다.

```text
/
├── bin
├── dev
├── etc
├── home
├── opt
├── proc
├── tmp
├── usr
└── var
```

여기서 `/`는:

> Linux filesystem의 가장 위

라고 생각하면 된다.

---

# 7. `/home`

일반 사용자의 개인 directory가 위치한다.

예를 들어 username이 `daye`라면:

```text
/home/daye
```

가 home directory가 될 수 있다.

```text
/home
├── daye
├── robot
└── developer
```

Shell에서는 자신의 home directory를:

```text
~
```

로 표현할 수 있다.

즉:

```text
~
```

는 예를 들어:

```text
/home/daye
```

와 같은 의미가 된다.

그래서:

```bash
cd ~
```

를 실행하면 home directory로 이동한다.

그냥:

```bash
cd
```

를 실행해도 보통 home으로 이동한다.

---

# 8. 현재 위치 확인: `pwd`

`pwd`는:

**Print Working Directory**

이다.

```bash
pwd
```

예:

```text
/home/daye/vision60_ws
```

현재 shell이 filesystem의 어디에 있는지 알려준다.

---

# 9. 파일 보기: `ls`

현재 directory의 파일을 확인:

```bash
ls
```

자세한 정보:

```bash
ls -l
```

숨김 파일까지:

```bash
ls -la
```

여기서 `-l`, `-a` 같은 것을 **option**이라고 한다.

```text
ls
│
└── command

-l
│
└── option
```

`-a`는 hidden file까지 보여준다.

Linux에서 이름이 `.`으로 시작하는 파일은 일반적으로 hidden file이다.

예:

```text
.bashrc
.git
.ssh
```

---

# 10. Absolute Path와 Relative Path

Linux path에는 두 가지 중요한 형태가 있다.

## Absolute Path

Root `/`에서 시작하는 전체 경로.

```text
/home/daye/vision60_ws/src
```

항상 `/`로 시작한다.

---

## Relative Path

현재 directory를 기준으로 하는 경로.

현재 위치가:

```text
/home/daye/vision60_ws
```

이고:

```bash
cd src
```

를 실행하면:

```text
/home/daye/vision60_ws/src
```

로 이동한다.

---

# 11. `.`과 `..`

Linux에서:

```text
.
```

은 **현재 directory**를 의미한다.

```text
..
```

은 **부모 directory**를 의미한다.

예:

```bash
cd ..
```

현재:

```text
/home/daye/vision60_ws/src
```

였다면:

```text
/home/daye/vision60_ws
```

로 이동한다.

---

# 12. `/etc`

`/etc`에는 system configuration 파일이 많이 존재한다.

예:

```text
/etc/ssh/
/etc/systemd/
/etc/hosts
```

즉:

```text
/etc
    ↓
System Configuration
```

정도로 기억하면 된다.

로봇에서 network, SSH, system service 등을 설정하다 보면 자주 만나게 된다.

---

# 13. `/dev`

`/dev`는 매우 중요하다.

**Device file**들이 위치한다.

Linux에서는 많은 hardware device를 파일처럼 표현한다.

예:

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/video0
```

예를 들어 USB serial 장치를 연결하면:

```text
Sensor
  │
 USB
  │
  ▼
Linux Kernel
  │
  ▼
/dev/ttyUSB0
```

처럼 나타날 수 있다.

Camera는:

```text
/dev/video0
```

등으로 나타날 수 있다.

그래서 로봇에서 sensor driver를 다룰 때 `/dev`를 자주 보게 된다.

---

# 14. `/proc`

`/proc`는 일반적인 disk directory와 조금 다르다.

Kernel과 실행 중인 process 정보를 보여주는 **virtual filesystem**이다.

예:

```text
/proc/cpuinfo
/proc/meminfo
```

확인:

```bash
cat /proc/cpuinfo
```

```bash
cat /proc/meminfo
```

즉 실제 SSD에 CPU 정보가 저장되어 있는 것이 아니라,
Kernel이 현재 system 정보를 filesystem 형태로 제공한다고 생각하면 된다.

---

# 15. `/opt`

`/opt`는 optional/add-on software를 설치할 때 자주 사용된다.

ROS 2를 사용하면 아주 익숙한 경로가 있다.

```text
/opt/ros/humble
```

예:

```bash
source /opt/ros/humble/setup.bash
```

즉 ROS 2 Humble이 system-wide로 설치되어 있다면 관련 파일들이:

```text
/opt/ros/humble
```

아래 존재할 수 있다.

---

# 16. `/usr`

`/usr`에는 user-space program, library, header 등이 많이 들어간다.

예:

```text
/usr/bin
/usr/lib
/usr/include
```

명령어를 찾다 보면:

```text
/usr/bin/python3
/usr/bin/git
```

같은 경로를 볼 수 있다.

---

# 17. `/var`

`/var`는 **variable data**를 저장하는 곳이다.

실행 중 계속 변하는 데이터가 많이 들어간다.

대표적으로:

```text
/var/log
```

에 system log가 저장된다.

Service 문제를 디버깅할 때 `/var/log`를 확인하는 경우가 많다.

---

# 18. Process란?

프로그램 파일과 실행 중인 프로그램은 다르다.

예를 들어 disk에:

```text
FAST-LIO2 executable
```

이 존재한다고 하자.

아직 실행하지 않았다면 그냥 **program**이다.

실행하면:

```text
Program
   │
   │ execute
   ▼
Process
```

가 된다.

즉:

> **Process = 실행 중인 program의 instance**

이다.

---

# 19. PID

Linux는 각 process에 **PID(Process ID)**를 부여한다.

예:

```text
PID 1423 → fastlio_mapping
PID 1502 → rviz2
PID 1601 → velodyne_driver
```

확인:

```bash
ps
```

더 많은 process:

```bash
ps aux
```

특정 process 검색:

```bash
ps aux | grep fastlio
```

---

# 20. `top`과 `htop`

실행 중인 process와 CPU/RAM 사용량을 실시간으로 확인할 수 있다.

```bash
top
```

조금 더 보기 편한 도구로:

```bash
htop
```

도 많이 사용한다.

예를 들어 SLAM이 Jetson의 CPU를 얼마나 사용하는지 확인할 때 유용하다.

```text
FAST-LIO2
    │
    ├── CPU usage
    └── RAM usage
```

---

# 21. Process 종료: `kill`

process를 종료할 때 PID를 사용할 수 있다.

```bash
kill 1423
```

기본적으로 process에게 종료 signal을 보낸다.

강제 종료:

```bash
kill -9 1423
```

하지만 `kill -9`는 process가 cleanup할 기회를 주지 않고 강제로 종료하므로
무조건 첫 번째 선택으로 사용하는 것은 좋지 않다.

보통:

```bash
kill PID
```

를 먼저 시도한다.

---

# 22. Foreground와 Background

Terminal에서:

```bash
ros2 launch ...
```

를 실행하면 일반적으로 foreground에서 실행된다.

즉 terminal을 해당 process가 점유한다.

```text
Terminal
   │
   └── ros2 launch
```

`Ctrl + C`를 누르면 일반적으로 interrupt signal을 보내 종료할 수 있다.

명령 끝에 `&`를 붙이면 background로 실행할 수 있다.

```bash
my_program &
```

---

# 23. Parent Process와 Child Process

Process는 다른 process를 만들 수 있다.

예:

```text
bash
 │
 └── ros2 launch
       │
       ├── lidar_driver
       ├── fastlio
       └── rviz2
```

Shell에서 프로그램을 실행하면 shell이 parent가 되고,
실행된 프로그램이 child process가 되는 구조를 볼 수 있다.

---

# 24. `exec`는 무엇인가?

Bash에서:

```bash
exec some_program
```

을 실행하면 새로운 child process를 단순히 하나 추가하는 것과 다르다.

현재 shell process가 `some_program`으로 **교체**된다.

예:

```bash
echo "before"

exec ros2 launch my_package bringup.launch.py

echo "after"
```

구조는:

```text
bash
 │
 ├── echo before
 │
 └── exec
       ↓
bash process 자체가
ros2 launch process로 교체
```

따라서 `ros2 launch`가 끝난다고 해도 원래 bash script로 돌아와:

```text
echo "after"
```

를 계속 실행하는 구조가 아니다.

그래서 `after`가 실행되지 않는다.

---

# 25. Service란?

로봇에는 부팅할 때 자동으로 실행되어야 하는 프로그램이 있을 수 있다.

예:

```text
network daemon
robot daemon
sensor service
```

이런 장기 실행 프로그램을 Linux에서는 service로 관리하는 경우가 많다.

Ubuntu에서는 주로:

```text
systemd
```

가 service를 관리한다.

---

# 26. systemd와 systemctl

Service 상태 확인:

```bash
systemctl status <service>
```

시작:

```bash
sudo systemctl start <service>
```

종료:

```bash
sudo systemctl stop <service>
```

재시작:

```bash
sudo systemctl restart <service>
```

부팅할 때 자동 실행:

```bash
sudo systemctl enable <service>
```

자동 실행 해제:

```bash
sudo systemctl disable <service>
```

구조를 단순화하면:

```text
systemd
  │
  ├── ssh.service
  ├── networking.service
  └── robot.service
```

라고 생각할 수 있다.

---

# 27. Process와 Service의 차이

Service도 결국 process로 실행된다.

차이는 **관리 방식**에 있다.

```text
Process
→ 실행 중인 프로그램

Service
→ systemd 등에 의해
  지속적으로 관리되는 프로그램
```

예를 들어 직접:

```bash
./robot_program
```

을 실행할 수도 있지만,

systemd service로 등록하면:

```bash
sudo systemctl start robot
```

처럼 관리할 수 있다.

---

# 28. Permission

Linux에서는 모든 사용자가 모든 파일을 수정할 수 있는 것이 아니다.

`ls -l`을 실행하면:

```text
-rwxr-xr-x
```

같은 것을 볼 수 있다.

대략:

```text
r = read
w = write
x = execute
```

이다.

그리고 permission은:

```text
Owner
Group
Others
```

기준으로 나뉜다.

---

# 29. `sudo`

`sudo`를 매우 자주 사용하지만 정확히 이해하는 것이 중요하다.

예:

```bash
sudo apt update
```

`sudo`는 허가된 사용자가 명령을 **다른 사용자(기본적으로 root)의 권한으로 실행**할 수 있게 한다.

Root user는 system에 매우 강력한 권한을 가진 관리자 계정이다.

그래서:

```bash
sudo
```

를 무조건 붙이는 습관은 좋지 않다.

특히:

```bash
sudo rm ...
```

같은 명령은 system 파일까지 삭제할 수 있으므로 주의해야 한다.

---

# 30. `chmod`

파일 permission을 변경한다.

예:

```bash
chmod +x start.sh
```

의 의미:

```text
chmod
  │
  └── change mode

+x
  │
  └── execute permission 추가
```

즉:

```bash
./start.sh
```

처럼 script를 직접 실행할 수 있도록 execute permission을 추가할 때 자주 사용한다.

---

# 31. Package Manager와 `apt`

Ubuntu에서는 software package를 관리할 때 `apt`를 많이 사용한다.

Package 목록 업데이트:

```bash
sudo apt update
```

Package 설치:

```bash
sudo apt install git
```

Package 제거:

```bash
sudo apt remove git
```

중요:

```bash
apt update
```

는 모든 프로그램을 최신 버전으로 update한다는 뜻이 아니다.

Package repository에서:

> 어떤 package version을 설치할 수 있는지 목록 정보를 갱신

하는 작업이다.

실제 package upgrade는:

```bash
sudo apt upgrade
```

등으로 별도로 수행한다.

---

# 32. Environment Variable

Environment Variable은 process가 실행될 때 참고할 수 있는 **이름-값 형태의 환경 정보**다.

가장 중요한 점은:

> **환경변수는 컴퓨터 전체에 하나로 존재하는 전역 변수가 아니라, 각 process가 자기 자신의 environment를 가진다.**

즉, 실행 중인 Bash shell 하나도 하나의 process다.

```text
Terminal
   │
   ▼
bash process
PID = 1234

Environment:
├── HOME=/home/user
├── PATH=/usr/bin:/bin
└── ROS_DOMAIN_ID=123
```

따라서 Bash에서:

```bash
export ROS_DOMAIN_ID=123
```

을 실행한다는 것은 개념적으로:

> **현재 Bash process(PID)의 environment에 `ROS_DOMAIN_ID=123`이라는 값을 설정한다.**

라고 이해할 수 있다.

즉:

```text
Bash A
PID = 1234

ROS_DOMAIN_ID=123
```

이라고 해서 다른 Bash process의 environment가 자동으로 변경되는 것은 아니다.

예를 들어 Terminal을 두 개 열었다고 하자.

```text
Terminal 1
└── Bash A
    PID = 1234
    ROS_DOMAIN_ID=123


Terminal 2
└── Bash B
    PID = 5678
    ROS_DOMAIN_ID 없음
```

Bash A에서:

```bash
export ROS_DOMAIN_ID=123
```

을 실행해도 Bash B에는 자동으로 적용되지 않는다.

즉 핵심은:

```text
Bash 1개
=
Process 1개
=
고유한 PID
=
자기 자신의 Environment
```

이다.

환경변수는 기본적으로 **process 단위로 관리된다.**

---

## Environment Variable 확인

예:

```bash
echo $HOME
```

```bash
echo $PATH
```

ROS 2에서는:

```bash
echo $ROS_DOMAIN_ID
```

같은 것을 볼 수 있다.

예:

```bash
export ROS_DOMAIN_ID=123
```

그러면 현재 Bash process에 해당 환경변수가 설정되고,
이 Bash에서 새롭게 실행되는 child process들이 이 값을 상속받을 수 있다.

```text
bash
PID = 1234

Environment:
ROS_DOMAIN_ID=123
      │
      │ child process 생성
      ▼
ros2
PID = 2000

Environment:
ROS_DOMAIN_ID=123
```

즉 환경변수는:

```text
Parent Process
      │
      │ Environment 상속
      ▼
Child Process
```

방향으로 전달될 수 있다.

---

## Child Process는 자신의 Environment를 가진다

중요한 점은 child process가 부모의 environment 자체를 공유하는 것이 아니라,
자신의 environment를 갖는다는 것이다.

개념적으로:

```text
bash
PID=100
ROS_DOMAIN_ID=123
      │
      ▼
python
PID=200
ROS_DOMAIN_ID=123
```

이후 Python에서:

```python
import os

os.environ["ROS_DOMAIN_ID"] = "456"
```

으로 변경한다고 하자.

그러면:

```text
bash
PID=100
ROS_DOMAIN_ID=123

        │

        └── python
            PID=200
            ROS_DOMAIN_ID=456
```

가 된다.

Python이 자신의 환경변수를 변경했다고 해서
부모 Bash의 값까지 `456`으로 바뀌는 것은 아니다.

즉 환경변수 전달은 기본적으로:

```text
Parent
   ↓
Child
   ↓
Grandchild
```

방향으로 이루어진다.

Child가 자신의 environment를 변경한다고 해서:

```text
Child
   ↑
Parent
```

방향으로 부모의 environment가 변경되지는 않는다.

---

# 33. `export`

Bash에서:

```bash
MY_VALUE=123
```

과:

```bash
export MY_VALUE=123
```

은 차이가 있다.

먼저:

```bash
MY_VALUE=123
```

은 Bash가 관리하는 **shell variable**을 만든다.

```text
bash process

Shell Variable:
MY_VALUE=123
```

현재 Bash에서는:

```bash
echo $MY_VALUE
```

로 사용할 수 있다.

하지만 이 상태에서는 일반적으로 새로 실행하는 child process의 environment로 전달되지 않는다.

---

반면:

```bash
export MY_VALUE=123
```

을 하면 해당 변수를 **environment variable로 내보내서 앞으로 생성되는 child process가 상속할 수 있게 한다.**

즉:

```text
bash
PID=100

MY_VALUE=123
(exported)
      │
      ├──────────────┐
      ▼              ▼
   Python           ros2
   PID=200          PID=300

MY_VALUE=123      MY_VALUE=123
```

이 된다.

`export`는 다음처럼 나눠서 써도 된다.

```bash
MY_VALUE=123
export MY_VALUE
```

또는 한 번에:

```bash
export MY_VALUE=123
```

으로 작성할 수 있다.

---

## ROS 2 Example

예:

```bash
export ROS_DOMAIN_ID=123
```

후:

```bash
ros2 run package_a node_a
```

그리고:

```bash
ros2 run package_b node_b
```

를 실행한다면:

```text
bash
PID=100

ROS_DOMAIN_ID=123
      │
      ├── ros2 node A
      │      └── ROS_DOMAIN_ID=123
      │
      └── ros2 node B
             └── ROS_DOMAIN_ID=123
```

처럼 child process들이 값을 상속받을 수 있다.

그래서 ROS 2 설정에서 `export`를 자주 본다.

예:

```bash
export ROS_DOMAIN_ID=123
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=$HOME/cyclonedds.xml
```

이후 실행되는 ROS 2 process들이 이 환경 정보를 참고할 수 있다.

---

## Python subprocess도 같은 원리다

예를 들어:

```bash
export ROS_DOMAIN_ID=123
python3 start_robot.py
```

를 실행하면:

```text
Bash
PID=100
ROS_DOMAIN_ID=123
      │
      ▼
Python
PID=200
ROS_DOMAIN_ID=123
```

Python이 다시:

```python
subprocess.run(["ros2", "node", "list"])
```

을 실행하면:

```text
Bash
PID=100
ROS_DOMAIN_ID=123
      │
      ▼
Python
PID=200
ROS_DOMAIN_ID=123
      │
      ▼
ros2
PID=300
ROS_DOMAIN_ID=123
```

처럼 이어서 상속될 수 있다.

---

## 핵심 Mental Model

```text
Bash 1개
=
Process 1개
=
PID 1개
=
자기 자신의 Environment
```

그리고:

```text
export VAR=value
```

은:

```text
현재 Bash process에 환경변수 설정
             +
앞으로 생성할 child process가
상속받을 수 있도록 export
```

한다고 이해하면 된다.

전체 구조:

```text
bash (PID 100)
Environment:
ROS_DOMAIN_ID=123
        │
        ├───────────────┐
        ▼               ▼
python (PID 200)    ros2 (PID 300)
ROS_DOMAIN_ID=123   ROS_DOMAIN_ID=123
        │
        ▼
subprocess (PID 400)
ROS_DOMAIN_ID=123
```

따라서 **환경변수는 시스템 전체의 전역 설정이 아니라 process별 환경이며, parent process에서 child process로 상속될 수 있는 설정 정보**라고 이해하는 것이 가장 중요하다.

---

# 34. `$PATH`

`PATH`는 매우 중요한 environment variable이다.

확인:

```bash
echo $PATH
```

예:

```text
/usr/local/bin:/usr/bin:/bin
```

Shell에서:

```bash
python3
```

라고 입력했을 때 우리는:

```bash
/usr/bin/python3
```

라고 전체 경로를 입력하지 않는다.

Shell이 `$PATH`에 등록된 directory들을 검색하기 때문이다.

```text
python3
   │
   ▼
Search PATH
   │
   ├── /usr/local/bin
   ├── /usr/bin   ← 발견
   └── /bin
```

어떤 executable이 실행되는지 확인:

```bash
which python3
```

---

# 35. `source`는 무엇인가?

ROS 2를 사용하면 매우 자주 본다.

```bash
source /opt/ros/humble/setup.bash
```

이 명령은 `setup.bash`를 **현재 shell 안에서 실행**한다.

왜 중요한가?

그 script가 현재 shell의 environment variable을 변경해야 하기 때문이다.

예:

```text
Before source

PATH
PYTHONPATH
AMENT_PREFIX_PATH
...

      │
      │ source setup.bash
      ▼

After source

ROS 2 관련 경로 추가
```

그래서 ROS 2 명령어와 package를 찾을 수 있게 된다.

---

# 36. 그냥 실행하는 것과 `source`의 차이

다음처럼 별도 process에서 script를 실행하면:

```bash
bash setup.bash
```

대략:

```text
Current Shell
     │
     └── New Bash
            │
            └── environment 변경
```

새 Bash가 끝나면 그 environment 변경도 사라진다.

반면:

```bash
source setup.bash
```

는:

```text
Current Shell
     │
     └── environment 직접 변경
```

이다.

그래서 ROS에서 `source`가 중요하다.

bash setup.bash
→ 새로운 Bash process에서 실행
→ 부모 Bash 환경은 안 바뀜

source setup.bash
→ 현재 Bash process에서 실행
→ 현재 Bash 환경이 바뀜

---

# 37. `.bashrc`

Bash를 열 때 자동으로 실행되는 사용자 설정 파일 중 하나가:

```text
~/.bashrc
```

이다.

예를 들어 매번:

```bash
source /opt/ros/humble/setup.bash
```

를 입력하기 귀찮다면 `.bashrc`에 넣는 경우가 있다.

```bash
source /opt/ros/humble/setup.bash
```

또는:

```bash
export ROS_DOMAIN_ID=123
```

등을 넣을 수 있다.

하지만 `.bashrc`에 너무 많은 설정을 넣으면
나중에 환경 충돌 원인을 찾기 어려울 수 있으므로 무엇을 추가했는지 이해하는 것이 중요하다.

---

# 38. SSH란?

Vision60이나 Jetson을 다루면서 매우 자주 사용하는 것이 SSH다.

SSH는:

**Secure Shell**

이다.

예:

```bash
ssh daye@192.168.0.18
```

구조:

```text
My Laptop
    │
    │ Network
    │ SSH
    ▼
Jetson
192.168.0.18
    │
    ▼
Remote Shell
```

즉 내 Mac에서 Jetson의 shell을 원격으로 사용하는 것이다.

---

# 39. SSH한다고 프로그램이 Mac에서 실행되는 것은 아니다

이 부분이 중요하다.

Mac에서:

```bash
ssh daye@jetson
```

한 뒤:

```bash
ros2 launch ...
```

를 실행했다면 실제 프로그램은:

```text
Mac
 │
 │ SSH input/output
 ▼
Jetson
 │
 └── ros2 launch
```

즉 **Jetson CPU에서 실행된다.**

Mac은 terminal을 통해 명령을 보내고 결과를 보고 있을 뿐이다.

Chapter 2에서 배운 architecture와 연결하면:

```text
Mac ARM64
    │
    │ SSH
    ▼
Jetson ARM64
```

이어도 각각 독립된 컴퓨터다.

---

# 40. SSH와 Network

SSH를 사용하려면 두 컴퓨터 사이에 network route가 존재해야 한다.

예:

```text
Mac
192.168.0.10
      │
      │ Wi-Fi / Ethernet
      ▼
Jetson
192.168.0.18
```

두 장치가 통신할 수 있다면 SSH 연결이 가능할 수 있다.

그래서 로봇 Wi-Fi, 회사 Wi-Fi, Ethernet, IP address 개념이 중요해진다.

이 부분은 이후 Networking Chapter에서 더 자세히 다룬다.

---

# 41. Vision60 Bringup Script를 Linux 관점에서 보기

예를 들어 이런 script가 있다고 하자.

```bash
#!/usr/bin/env bash

set -e

WS="$HOME/vision60_ws"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

export ROS_DOMAIN_ID=123

mkdir -p "$WS/logs"

exec ros2 launch vision60_bringup bringup.launch.py
```

이제 각 줄이 무엇인지 연결할 수 있다.

```text
#!/usr/bin/env bash
→ 이 script를 Bash로 실행

set -e
→ command 실패 시 script 종료

WS="$HOME/vision60_ws"
→ shell variable 생성

source ...
→ 현재 shell의 ROS environment 설정

export ROS_DOMAIN_ID=123
→ child process에 전달할 environment variable

mkdir -p
→ directory 생성

exec ros2 launch ...
→ 현재 shell process를 ros2 launch로 교체
```

즉 이전에는 단순히 "복잡한 shell script"처럼 보였던 코드가
Linux의 몇 가지 기본 개념 조합이라는 것을 알 수 있다.

---

# 42. Linux에서 로봇 프로그램이 실행되는 전체 흐름

Vision60 + Jetson 환경을 단순화하면:

```text
Power ON
   │
   ▼
Linux Kernel
   │
   ▼
Ubuntu
   │
   ├── systemd
   │     ├── network
   │     └── ssh
   │
   ▼
User Login / SSH
   │
   ▼
Bash
   │
   ├── source ROS 2
   ├── export environment
   │
   ▼
ros2 launch
   │
   ├── LiDAR Driver
   ├── IMU Driver
   ├── FAST-LIO2
   └── Navigation
```

이 구조가 앞으로 Jetson을 디버깅할 때 기본적인 mental model이 된다.

---

# 43. 실무에서 자주 쓰는 명령어

| 목적 | Command |
|---|---|
| 현재 위치 | `pwd` |
| 파일 목록 | `ls -la` |
| directory 이동 | `cd` |
| directory 생성 | `mkdir -p dir` |
| 파일 내용 | `cat file` |
| CPU architecture | `uname -m` |
| CPU 정보 | `lscpu` |
| Process 확인 | `ps aux` |
| 실시간 resource | `top` / `htop` |
| Process 종료 | `kill PID` |
| 실행 파일 위치 | `which command` |
| Environment 확인 | `env` |
| 변수 확인 | `echo $VAR` |
| Service 상태 | `systemctl status` |
| IP 확인 | `ip addr` |
| SSH | `ssh user@IP` |
| Disk 사용량 | `df -h` |
| Directory 크기 | `du -sh directory` |

---

# 44. Mini Practice

Jetson 또는 Ubuntu Linux에서 다음을 직접 실행해본다.

```bash
pwd
```

```bash
echo $HOME
```

```bash
echo $SHELL
```

```bash
echo $PATH
```

```bash
which python3
```

```bash
uname -m
```

```bash
ps aux | head
```

```bash
cat /proc/meminfo | head
```

```bash
ls /dev | head
```

ROS 2가 설치되어 있다면:

```bash
ls /opt/ros
```

그리고:

```bash
source /opt/ros/humble/setup.bash
```

전후로:

```bash
env | grep ROS
```

를 비교해본다.

---

# 45. 오늘의 핵심

Linux를 처음 공부할 때 모든 command를 외울 필요는 없다.

다음 구조를 이해하는 것이 더 중요하다.

```text
Hardware
   ↑
Linux Kernel
   ↑
Ubuntu
   ↑
Process / Service
   ↑
Shell
   ↑
ROS 2 / Robot Software
```

그리고 다음 개념은 반드시 구분한다.

```text
Program ≠ Process

Terminal ≠ Shell

Linux ≠ Ubuntu

Process ≠ Service

PATH ≠ 현재 Directory

source script
≠
bash script

SSH 접속
≠
내 컴퓨터에서 remote program 실행
```

---

# 46. Robot Debugging Mental Model

로봇에서 프로그램이 안 돌아갈 때 무작정 코드를 수정하기 전에 어느 층의 문제인지 생각한다.

```text
Application
FAST-LIO2 / ROS 2
        │
        ▼
Environment
PATH / ROS_DOMAIN_ID
        │
        ▼
Process / Service
        │
        ▼
Linux
Permission / Device
        │
        ▼
Network
IP / Interface
        │
        ▼
Hardware
LiDAR / IMU / Jetson
```

예를 들어 LiDAR가 안 들어온다고 해서 반드시 FAST-LIO2 문제인 것은 아니다.

```text
LiDAR hardware?
      ↓
/dev device?
      ↓
driver process?
      ↓
ROS 2 topic?
      ↓
FAST-LIO2?
```

처럼 아래부터 확인하면 문제를 훨씬 빠르게 좁힐 수 있다.

---

# Next Chapter

## Chapter 4. NVIDIA Jetson & JetPack

다음 Chapter에서는 지금까지 배운 내용을 실제 Jetson architecture에 연결한다.

- Jetson은 정확히 무엇인가?
- Jetson Module과 Developer Kit은 무엇이 다른가?
- Xavier, Orin, Thor는 무엇이 다른가?
- JetPack은 OS인가?
- JetPack, Ubuntu, L4T의 관계는?
- CUDA란 무엇인가?
- cuDNN과 TensorRT는 무엇인가?
- Jetson의 CPU와 GPU는 RAM을 어떻게 사용하는가?
- Unified Memory는 무엇인가?
- `nvidia-smi`가 Jetson에서는 왜 PC와 다를 수 있는가?
- `tegrastats`는 무엇인가?
- Jetson에서 power mode와 thermal throttling이 왜 중요한가?

Chapter 4부터는 지금까지의:

```text
CPU / GPU / RAM
       +
ARM64
       +
Linux
```

를 합쳐서 실제 **NVIDIA Jetson이라는 Edge Computer가 어떻게 구성되는지** 살펴본다.