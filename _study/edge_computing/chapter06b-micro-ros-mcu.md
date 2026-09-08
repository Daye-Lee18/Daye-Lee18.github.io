---
title: "Chapter 6.5. micro-ROS와 MCU — /mcu/... 토픽은 어디서 오는가"
importance: 7.5
---

> **Goal:** 로봇 안에서 `/mcu/state/imu` 같은 토픽이 **어느 칩에서 출발해 어떤 경로로**
> ROS 2 그래프에 도착하는지 설명하고, micro-ROS·CAN·Serial이 각각
> 소프트웨어 계층인지 하드웨어 통로인지 구분할 수 있다.

Chapter 6은 ROS 2를 메인 컴퓨터 위에서만 봤다.
그런데 실제 로봇의 토픽 목록을 보면 `/mcu/...`로 시작하는 것들이 잔뜩 나온다.
그건 Linux가 아니라 **RAM이 수십 KB뿐인 칩**에서 오는 데이터다.
이 chapter는 그 경계를 다룬다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `ros2 run micro_ros_agent micro_ros_agent serial ...` · `ls -l /dev/ttyACM*` · `ros2 topic hz`

| 명령어                                                                         | 하는 일                                     |
| :----------------------------------------------------------------------------- | :------------------------------------------ |
| `ls -l /dev/ttyACM*` / `ls -l /dev/ttyUSB*`                                    | MCU가 어느 device node로 잡혔는지           |
| `dmesg -w`                                                                     | MCU를 꽂는 순간 커널이 뭐라고 하는지        |
| `sudo usermod -aG dialout $USER`                                               | serial 권한 (재로그인 필요)                 |
| `ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200` | **Agent 실행 — 이게 없으면 토픽이 안 뜬다** |
| `ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888`                    | Ethernet/Wi-Fi transport일 때               |
| `ros2 node list`                                                               | MCU가 ROS 2 노드로 올라왔는지               |
| `ros2 topic hz /mcu/state/imu`                                                 | **실제** 발행 주기 (MCU가 살아 있는지)      |
| `ros2 topic echo /mcu/state/imu --once`                                        | 데이터가 진짜 오는지                        |
| `ros2 topic type /mcu/state/imu`                                               | 메시지 타입                                 |
| `ip link show can0`                                                            | CAN 인터페이스 상태                         |
| `sudo ip link set can0 up type can bitrate 1000000`                            | CAN 올리기                                  |
| `candump can0`                                                                 | CAN 프레임 실시간 (can-utils)               |

---

# 1. `/mcu/...` 토픽은 어디서 오는가

Vision60 같은 로봇의 토픽 목록에는 이런 것들이 섞여 있다.

```text
/mcu/state/imu            ← MCU에서 온다
/mcu/state/motorData      ← MCU에서 온다
/mcu/state/battery        ← MCU에서 온다
/mcu/command/manual_twist ← MCU로 간다

/gx5/imu/data             ← 메인 컴퓨터에 붙은 센서 드라이버
/scan                     ← LiDAR 드라이버
```

`/mcu/`라는 이름이 붙은 것은 대부분 **메인 컴퓨터가 만든 데이터가 아니다.**
모터 옆, 배터리 옆, IMU 옆에 붙어 있는 별도의 작은 칩이 만든다.

---

# 2. MCU도 컴퓨터다

MCU(Microcontroller Unit)는 CPU 코어, 메모리, 입출력을 **칩 하나에 다 넣은 초소형 컴퓨터**다.
흔히 원칩 컴퓨터라고 부른다.

```text
             MCU 하나 안에

  ┌────────────────────────────────────┐
  │  CPU 코어      명령어 실행          │
  │  Flash ROM     펌웨어가 저장됨       │
  │  RAM           수 KB ~ 수백 KB      │
  │  GPIO          핀 on/off            │
  │  ADC           아날로그 → 디지털     │
  │  UART/SPI/I2C/CAN   통신            │
  │  Timer / PWM   모터 제어             │
  └────────────────────────────────────┘
```

PC의 CPU는 연산만 하고 RAM·저장장치·GPU가 밖에 따로 있어야 하지만,
MCU는 동작에 필요한 것이 다 안에 들어 있다. 대신 규모가 작다.

---

# 3. MCU / MPU / SoC 구분

Chapter 1에서 SoC를 다뤘다. 여기에 MCU를 나란히 놓으면 경계가 분명해진다.

|           | MCU                             | MPU                | SoC                     |
| :-------- | :------------------------------ | :----------------- | :---------------------- |
| 예        | STM32, PIC, ATmega, ESP32       | 옛 방식의 CPU 단품 | Jetson Orin, 스냅드래곤 |
| 클럭      | 수 ~ 수백 MHz                   | 수백 MHz ~ GHz     | GHz, 멀티코어           |
| RAM       | 수 KB ~ 수 MB (내장)            | 외부 필요          | 수 GB (외부 DRAM)       |
| OS        | 없음(베어메탈) 또는 RTOS        | Linux              | Linux, Android          |
| 잘하는 일 | **정확한 타이밍의 실시간 제어** | 범용 연산          | 인지, SLAM, 딥러닝      |
| 로봇에서  | 모터·배터리·IMU 보드            | —                  | 메인 온보드 컴퓨터      |

`PIC`는 Microchip사의 MCU 제품군 이름이다. MCU와 별개 개념이 아니라 **MCU의 한 종류**다.

---

# 4. 왜 메인 컴퓨터가 다 하지 않고 MCU를 따로 두나

메인 컴퓨터가 훨씬 빠른데도 모터 제어를 맡기지 않는다. 이유는 속도가 아니라 **일정함**이다.

```text
메인 컴퓨터 (Linux)              MCU (베어메탈 / RTOS)
──────────────────────           ────────────────────────
멀티태스킹 OS                     펌웨어 하나만 반복
평균은 빠르다                     평균은 느리다
하지만 가끔 몇 ms 밀린다           대신 절대 안 밀린다

  1ms  1ms  1ms  8ms ← 여기         1ms  1ms  1ms  1ms  1ms
                 ↑
      로봇이 넘어지는 순간
```

4족 보행 로봇이 넘어지지 않으려면 제어 루프가 **매번 같은 주기로** 돌아야 한다.
평균 1 ms인데 가끔 8 ms인 것보다, 항상 정확히 2 ms인 편이 낫다.

그래서 역할을 나눈다.

```text
메인 컴퓨터   카메라·LiDAR 처리, SLAM, 경로 계획, 인지     (무겁고, 가끔 느려도 됨)
MCU          센서 읽기, 모터 토크 제어, 배터리 감시        (가볍고, 절대 밀리면 안 됨)
```

Chapter 15의 real-time 이야기가 여기서 하드웨어로 나타난 것이다.

---

# 5. MCU에서 ROS 2 토픽까지 — 데이터가 지나는 길

```text
 ① IMU 센서        물리적인 가속도·각속도
       │
 ② MCU            μs 단위로 읽고 가공. 이 구간은 ROS 2가 아니다
       │
 ③ 전선           UART / USB / CAN / Ethernet  ← 하드웨어 통로
       │
 ④ 메인 컴퓨터     받은 바이트를 ROS 2 메시지로 변환
       │
 ⑤ ROS 2 그래프    /mcu/state/imu 로 publish
```

핵심은 ③과 ④다. **③은 하드웨어 통로, ④는 소프트웨어 변환**이고,
이 둘을 어떻게 구성하느냐가 아래 세 가지 선택지다.

---

# 6. ROS 2와 micro-ROS

micro-ROS는 ④를 MCU 쪽으로 끌어당기는 방식이다.
MCU가 바이트를 던지고 마는 게 아니라, **자기가 직접 ROS 2 노드가 된다.**

![ROS 2 stack과 micro-ROS stack 비교](https://docs.vulcanexus.org/en/humble/_images/microros_stack.png)

<small>그림 출처: <a href="https://docs.vulcanexus.org/en/humble/rst/microros_documentation/index.html">Vulcanexus micro-ROS documentation</a> (eProsima)</small>

같은 내용을 글로 옮기면 이렇다.

```text
      ROS 2 (메인 컴퓨터)              micro-ROS (MCU)
   ──────────────────────           ──────────────────────
      User Application                User Application
              │                               │
        rclcpp / rclpy                      rclc          ← micro-ROS 전용
              │                               │
             rcl                             rcl          ← ROS 2에서 그대로 가져옴
              │                               │
             rmw                             rmw          ← ROS 2에서 그대로 가져옴
              │                               │
             DDS                    Micro XRCE-DDS Client ← micro-ROS 전용
              │                               │
            Linux                    RTOS (FreeRTOS,      ← 또는 베어메탈
                                      Zephyr, NuttX)
              │                               │
          x86 / ARM SoC                  마이크로컨트롤러
```

읽는 법이 중요하다.

- **가운데 `rcl`과 `rmw`는 양쪽이 똑같다.** ROS 2 것을 그대로 쓴다.
  그래서 MCU 위의 노드가 진짜 ROS 2 노드처럼 취급된다.
- **위아래만 갈아끼웠다.** 무거운 `rclcpp`(C++ 객체지향, 동적 할당) 대신
  가벼운 C 라이브러리 `rclc`를, 무거운 DDS 대신 `Micro XRCE-DDS Client`를 쓴다.
- 맨 아래는 Linux가 아니라 RTOS이거나 아예 OS가 없다.

---

# 7. micro-ROS Agent — 없으면 토픽이 안 뜬다

MCU는 DDS를 통째로 돌릴 수 없다. Discovery 하나만으로도 MCU의 RAM을 다 먹는다.
그래서 **무거운 일을 메인 컴퓨터가 대신 해주는 구조**를 쓴다. 그 대리인이 Agent다.

```text
   MCU                                메인 컴퓨터
 ┌────────────────────┐           ┌─────────────────────────────┐
 │ Micro XRCE-DDS     │  ──③──▶   │  micro-ROS Agent            │
 │ Client (가벼움)     │  전선     │   · DDS 전부 대행            │
 │                    │  ◀────    │   · discovery, graph 관리    │
 └────────────────────┘           │            │                 │
                                  │            ▼                 │
                                  │      진짜 DDS → ROS 2 그래프  │
                                  └─────────────────────────────┘
```

그래서 실무에서 가장 자주 겪는 증상이 이것이다.

```text
증상: MCU에 펌웨어도 잘 올라갔고 LED도 깜빡이는데
      ros2 topic list 에 아무것도 안 뜬다

원인: Agent를 안 띄웠다
```

Agent는 그냥 ROS 2 노드이므로 이렇게 실행한다.

```bash
sudo apt install ros-humble-micro-ros-agent

# Serial transport
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200

# UDP transport (Ethernet / Wi-Fi)
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888
```

Agent가 붙으면 그때부터 MCU의 노드가 `ros2 node list`에 나타난다.

---

# 8. micro-ROS는 소프트웨어, CAN·Serial은 하드웨어 통로

여기가 가장 헷갈리는 지점이다. **셋은 같은 층에 있는 선택지가 아니다.**

```text
        micro-ROS        ← 소프트웨어 미들웨어 (무엇을 어떤 형식으로 주고받나)
   ─────────────────────
   UART / USB / CAN / Ethernet / SPI / Wi-Fi
                         ← 하드웨어 통로 (바이트가 물리적으로 지나가는 길)
```

택배로 비유하면 이렇다.

```text
CAN, Serial   →  도로와 트럭        (물리적으로 실어 나르는 수단)
micro-ROS     →  택배 물류 시스템    (송장 양식, 주소 체계, 배송 추적)
```

micro-ROS는 자기 전선이 따로 있는 게 아니라 **기존 통로 위에 얹혀서 간다.**
micro-ROS 문서에서 이걸 transport agnostic이라고 부르고, 실제로 지원 목록이 이렇다.

```text
micro-ROS over Serial (UART / USB-CDC)   ← 가장 흔하다
micro-ROS over UDP / Ethernet / Wi-Fi
micro-ROS over CAN-FD
micro-ROS over SPI, Bluetooth, 6LoWPAN
```

그러니 **"CAN이냐 micro-ROS냐"는 원래 성립하지 않는 질문**이다.
현업에서 그렇게 말할 때 실제로 비교하는 것은 아래 두 가지 아키텍처다.

```text
방식 A — 직접 파싱
  MCU:     C 코드로 [헤더][데이터][CRC] 바이트 패킷을 직접 만든다
  전선:    UART 또는 CAN
  메인 PC: 그 바이트를 받아 쪼개고 검증하는 ROS 2 노드를 직접 짠다
  → 가볍고 빠르다. 대신 프로토콜과 파서를 양쪽에 직접 짜야 한다

방식 B — micro-ROS
  MCU:     rclc_publish() 를 그냥 호출한다
  전선:    UART 또는 CAN (똑같다)
  메인 PC: micro_ros_agent 만 띄운다
  → 파서를 안 짜도 된다. 대신 MCU의 메모리와 연산을 많이 먹는다
```

---

# 9. 세 가지를 한 표로

|                 | micro-ROS                    | CAN / CAN-FD                      | Serial (UART / USB-CDC)       |
| :-------------- | :--------------------------- | :-------------------------------- | :---------------------------- |
| 무엇인가        | 소프트웨어 미들웨어          | 하드웨어 버스                     | 하드웨어 점대점               |
| 계층            | 애플리케이션 (DDS-XRCE 기반) | 데이터 링크                       | 물리 / 데이터 링크            |
| 연결 형태       | Agent 경유                   | 버스 — 여러 대를 한 선에          | 1:1                           |
| 속도            | 아래 통로에 의존             | 1 Mbps (FD는 5~8 Mbps)            | 115 kbps ~ 수 Mbps (USB는 더) |
| MCU 자원        | **많이 먹음** (수십~수백 KB) | 매우 적음 (하드웨어가 처리)       | 적음                          |
| 실시간성        | Agent를 거쳐 보통            | **최상** (하드웨어 우선순위 중재) | 중간 (노이즈에 취약)          |
| ROS 2 연동      | **바로 노드가 됨**           | 브리지 노드를 직접 개발           | 파서 노드를 직접 개발         |
| 한 번에 보낼 양 | 제한 없음                    | 8 B (FD는 64 B)                   | 스트림이라 자유               |

---

# 10. 그래서 언제 무엇을 쓰나

```text
4족 로봇의 관절 모터 제어
  → CAN / CAN-FD
     선 두 가닥에 모터 드라이버 수십 개를 매달 수 있고,
     노이즈와 지연에 대한 보장이 필요하다

IMU 하나, 로봇 베이스 제어를 빨리 ROS 2에 붙이고 싶다
  → micro-ROS over Serial
     파서를 안 짜도 되니 프로토타이핑이 빠르다

1 kHz 이상 고주파 센서 스트리밍, MCU 자원이 빠듯하다
  → 바이너리 프로토콜 + Serial 직접 파싱
     미들웨어 오버헤드를 아예 없앤다
```

한 로봇 안에서 셋을 동시에 쓰는 것이 보통이다. 배타적인 선택이 아니다.

---

# 11. micro-ROS를 올리려면 MCU가 얼마나 되어야 하나

```text
현실적인 최소선   RAM  ~32 KB 이상, Flash ~100 KB 이상
편한 선          STM32F4 / F7 / H7, ESP32, Teensy 4.x 급
빠듯한 것        ATmega328 (아두이노 우노, RAM 2 KB) — 사실상 불가
```

RTOS 지원은 FreeRTOS, Zephyr, NuttX, Azure RTOS가 대표적이고,
STM32CubeIDE·ESP-IDF·Renesas e2 studio 같은 벤더 툴에도 빌드 지원이 들어가 있다.

주의할 점은 메모리보다 **제어 루프에 주는 영향**이다.
micro-ROS의 통신 처리가 MCU의 CPU 시간을 먹으므로,
1 kHz 모터 제어 루프와 같은 코어에서 돌리면 그 주기가 흔들릴 수 있다.

---

# 12. micro-ROS over Serial — 실제 순서

```bash
# ── 메인 컴퓨터 ──
# 1. MCU가 어느 device로 잡혔는지
dmesg -w                       # MCU를 꽂으면서 본다
ls -l /dev/ttyACM*             # 보통 /dev/ttyACM0

# 2. 권한 (한 번만, 재로그인 필요)
sudo usermod -aG dialout $USER

# 3. Agent 설치와 실행
sudo apt install ros-humble-micro-ros-agent
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200

# ── 다른 터미널 ──
# 4. MCU가 노드로 올라왔는지
ros2 node list
ros2 topic list

# 5. 데이터가 실제로 오는지
ros2 topic echo /mcu/state/imu --once
ros2 topic hz   /mcu/state/imu
```

Docker 안에서 Agent를 돌린다면 device와 네트워크를 넘겨야 한다.

```bash
docker run -it \
  --network host \
  --device /dev/ttyACM0 \
  -e ROS_DOMAIN_ID=42 \
  my-ros-image bash
```

`--device`로 좁히는 것이 `--privileged`보다 낫다. Chapter 9.5의 원칙 그대로다.

---

# 13. 안 될 때 보는 순서

```text
증상                              먼저 볼 것
────────────────────────────      ──────────────────────────────────
topic list 가 비어 있음            Agent를 띄웠나
                                  → ros2 run micro_ros_agent ...

Agent가 device를 못 염               권한과 경로
                                  → ls -l /dev/ttyACM0, dialout 그룹

Agent는 붙었는데 노드가 안 보임      MCU 펌웨어가 실제로 도는지
                                  → MCU 리셋, LED/디버그 UART 확인

노드는 보이는데 echo 가 멈춤        메시지 타입이나 QoS 불일치
                                  → ros2 topic type, ros2 topic info --verbose

토픽이 끊겼다 붙었다 함             baudrate 불일치, USB 케이블,
                                  MCU 제어 루프가 통신을 굶기고 있음

Docker 안에서만 안 보임             --network host 와 --device 를 줬나
                                  ROS_DOMAIN_ID 가 전달됐나
```

Chapter 10의 계층 좁히기를 MCU 경계에 적용한 것이다.

---

# 14. Mini Practice

```bash
# 1) MCU가 물리적으로 붙어 있나
lsusb
ls -l /dev/ttyACM* /dev/ttyUSB*

# 2) Agent 없이 topic list — 비어 있는 것이 정상이다
ros2 topic list

# 3) Agent를 띄우고 다시
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
#   다른 터미널에서
ros2 node list
ros2 topic hz /mcu/state/imu

# 4) Agent를 Ctrl-C 로 끄면 토픽이 사라지는지 확인
```

2번과 3번의 차이를 눈으로 보는 것이 이 chapter의 핵심이다.

---

# 15. 오늘의 핵심

```text
              하나의 로봇, 두 종류의 컴퓨터

  ┌──────────────────────────────────────────────┐
  │  메인 컴퓨터 (SoC + Linux)                    │
  │    SLAM, 인지, 경로 계획                       │
  │    ROS 2 + DDS + micro-ROS Agent             │
  └───────────────────┬──────────────────────────┘
                      │  UART / USB / CAN / Ethernet
                      │      ← 하드웨어 통로
  ┌───────────────────┴──────────────────────────┐
  │  MCU (STM32 등 + RTOS/베어메탈)               │
  │    센서 읽기, 모터 토크, 배터리 감시            │
  │    micro-ROS Client  또는  직접 만든 프로토콜   │
  └──────────────────────────────────────────────┘
```

---

# 16. 반드시 구분할 것

```text
micro-ROS (소프트웨어 미들웨어)
≠
CAN / Serial (하드웨어 통로)
   micro-ROS 는 이 위에 얹혀서 간다

DDS (메인 컴퓨터)
≠
Micro XRCE-DDS (MCU)

rclcpp (C++, 메인 컴퓨터)
≠
rclc (C, MCU)

rcl / rmw
=  양쪽이 같다. 갈아끼운 건 위아래뿐

micro-ROS Client (MCU)
≠
micro-ROS Agent (메인 컴퓨터)
   Agent 없이는 토픽이 안 뜬다

MCU (실시간, 밀리면 안 됨)
≠
SoC (연산, 가끔 밀려도 됨)

MCU ⊃ PIC
   PIC 는 Microchip 사의 MCU 제품군 이름
```

---

# 17. Chapter 연결

```text
Chapter 1
CPU / GPU / SoC — 하드웨어 단위

Chapter 5
Hardware Interfaces — UART, CAN, USB 물리 계층

Chapter 6
ROS 2 — node, topic, rcl, rmw, DDS

Chapter 6.5  ← 여기
micro-ROS — 그 stack을 MCU까지 내리면 어떻게 되나

Chapter 9.5
Docker — Agent를 컨테이너에서 돌릴 때 --device, --network host

Chapter 15
Real-Time — MCU를 따로 두는 진짜 이유
```

**참고 문서:** [Vulcanexus micro-ROS documentation](https://docs.vulcanexus.org/en/humble/rst/microros_documentation/index.html) ·
[micro-ROS 공식 사이트](https://micro.vulcanexus.org/)
