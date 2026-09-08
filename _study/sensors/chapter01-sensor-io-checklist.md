---
title: "Chapter 1. 센서를 붙일 때 확인할 7가지"
importance: 2
---

> **Goal:** 처음 보는 센서를 받았을 때 무엇을 어떤 순서로 확인해야 하는지 알고,
> 그 결과를 팀이 공유할 수 있는 형식으로 기록한다.

센서마다 회사도 다르고 프로토콜도 다르지만, **확인해야 할 항목은 항상 같다.**
그 목록을 먼저 정해 두면 새 센서를 붙일 때 헤매는 시간이 크게 줄어든다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `tcpdump` · `ros2 topic hz` · `dmesg -w`

| 명령어                                           | 하는 일                    |
| :----------------------------------------------- | :------------------------- |
| `ip addr` / `ip route`                           | 어느 인터페이스로 나가는지 |
| `ping -c 3 192.168.1.201`                        | 물리적으로 닿는지          |
| `sudo tcpdump -i eth1 -n -c 10 'udp port 2368'`  | **데이터가 실제로 오는지** |
| `sudo tcpdump -i eth1 -n 'host 192.168.1.201'`   | 그 센서와 오가는 전부      |
| `ss -tuln`                                       | 내 쪽 포트가 열려 있나     |
| `curl -s http://192.168.1.201/cgi/settings.json` | HTTP 설정 조회             |
| `lsusb` / `dmesg -w`                             | USB 센서 인식 확인         |
| `ls -l /dev/ttyUSB* /dev/video*`                 | device node와 권한         |
| `ros2 topic list`                                | 드라이버가 토픽을 냈나     |
| `ros2 topic hz /velodyne_points`                 | **실제 주기**              |
| `ros2 topic echo /x --once`                      | 실제 값                    |
| `ros2 topic type /x` + `ros2 interface show`     | 메시지 정의                |

---

# 1. 두 개의 통신이 따로 있다

가장 먼저 잡아야 할 그림이다. **설정과 데이터는 다른 길로 간다.**

```text
    호스트                                센서
   ┌────────┐                          ┌────────┐
   │        │ ──── HTTP / TCP ───────▶ │        │   설정·조회
   │  Thor  │                          │ LiDAR  │   (내가 물어본다)
   │        │ ◀─── UDP ──────────────  │        │   점군 데이터
   └────────┘                          └────────┘   (센서가 계속 쏜다)
```

이 구조를 알면 증상 해석이 달라진다.

```text
ping 은 되는데 점군이 없다
   → 네트워크는 연결됐다. UDP 경로나 목적지 설정 문제

curl 로 설정은 읽히는데 점군이 없다
   → 센서는 살아 있다. UDP 목적지 IP/포트가 내 쪽이 아닐 가능성

tcpdump 에는 패킷이 보이는데 ROS 토픽이 없다
   → 네트워크는 정상. 드라이버 설정 문제
```

**`tcpdump`에서 패킷이 보이느냐**가 네트워크 문제와 드라이버 문제를 가르는 경계선이다.

---

# 2. 항목 1 — 물리 인터페이스

```text
Ethernet     대부분의 3D LiDAR. 대역폭이 크고 거리가 길다
USB          카메라, 일부 IMU. 케이블 품질과 대역폭 공유가 문제
CAN          모터 드라이버, 배터리. 대역폭은 작지만 실시간성이 최상
Serial       저가 IMU, GNSS. 단순하고 확실
```

Edge Computing Chapter 5에서 다룬 그 계층이다. 확인은 이렇게 한다.

```bash
# Ethernet
ip addr                       # 랜 카드에 IP가 붙어 있나
ip route                      # 그 센서 대역이 어느 인터페이스로 나가나
ping -c 3 192.168.1.201

# USB
lsusb                         # 목록에 보이나
dmesg -w                      # 꽂는 순간 커널이 뭐라고 하나
ls -l /dev/video* /dev/ttyUSB*
```

**USB 센서는 `dmesg -w`를 켜 놓고 꽂는 것**이 가장 빠르다.
전원 부족, 케이블 불량, 드라이버 없음이 전부 여기 찍힌다.

---

# 3. 항목 2 — 설정 프로토콜

센서에 "FOV를 바꿔라", "회전 속도를 바꿔라"를 어떻게 말하는가.

```text
HTTP CGI        브라우저로 열리는 웹 UI가 있다. curl 로도 된다   (Velodyne)
TCP 전용 포트    회사가 정한 프로토콜. SDK 를 거쳐야 한다          (Hesai PTC)
JSON 설정 파일   드라이버가 읽어서 센서에 밀어 넣는다            (Livox)
SDK / API       C++/Python 라이브러리 호출                     (RealSense)
```

**설정을 바꾸면 확인하고, 필요하면 저장까지 해야 한다.**
전원을 껐다 켜면 되돌아가는 제품이 많다.

```bash
# 바꾸고
curl -X POST http://192.168.1.201/cgi/setting/fov --data 'start=0&end=359'

# 반드시 다시 읽어서 확인
curl -s http://192.168.1.201/cgi/settings.json | python3 -m json.tool
```

---

# 4. 항목 3·4 — 데이터 프로토콜과 포트

```text
UDP     대부분의 LiDAR. 빠르고 재전송이 없다. 놓치면 그냥 놓친다
TCP     일부 설정·상태. 순서와 도착이 보장된다
USB     카메라. 대역폭을 다른 USB 장치와 나눠 쓴다
```

LiDAR가 UDP를 쓰는 이유는 명확하다.

```text
점군은 초당 수십 MB 다
재전송해서 늦게 받은 스캔은 이미 쓸모가 없다
    → 놓치더라도 최신을 받는 편이 낫다
```

대신 **패킷을 놓쳐도 아무도 알려주지 않는다.** 그래서 `ros2 topic hz`로 실제 주기를 재는 습관이 필요하다.

포트는 반드시 실측한다.

```bash
sudo tcpdump -i eth1 -n -c 10 'udp port 2368'
```

```text
· 패킷이 보인다        → 센서가 내 쪽으로 쏘고 있다
· 아무것도 안 보인다    → 포트가 다르거나, 목적지가 내가 아니거나, 방화벽
```

포트를 모를 때는 그 호스트와 오가는 것을 전부 본다.

```bash
sudo tcpdump -i eth1 -n 'host 192.168.1.201'
```

`tcpdump`는 **호스트에서** 돌리는 것이 안전하다.
컨테이너 안에서는 `sudo`가 비밀번호를 요구해 막히는 경우가 많다.

```text
sudo: a password is required
```

---

# 5. 항목 5·6·7 — 드라이버와 토픽

```text
제조사 ROS 드라이버      UDP/USB 를 받아 ROS 메시지로 바꾼다
raw 토픽                드라이버가 처음 뱉는 것. 원본 패킷에 가깝다
출력 토픽               실제로 쓰는 PointCloud2 / Image / Imu
```

Velodyne을 예로 보면 두 단계다.

```text
UDP 2368  →  velodyne_driver  →  /velodyne_packets   (raw)
                                       │
                             velodyne_pointcloud
                                       ▼
                                 /velodyne_points    (PointCloud2)
```

**둘을 구분해야 어디까지 왔는지 알 수 있다.**

```text
/velodyne_packets 만 있다   → 네트워크는 OK, 변환 노드가 안 떴다
둘 다 없다                  → 드라이버 자체가 안 떴거나 UDP 를 못 받는다
둘 다 있는데 hz 가 낮다     → 패킷 손실. MTU, 네트워크 부하 확인
```

확인 순서는 항상 같다.

```bash
ros2 topic list
ros2 topic hz   /velodyne_points     # 실제 주기 (10 Hz 나오나)
ros2 topic echo /velodyne_points --once
ros2 topic type /velodyne_points     # sensor_msgs/msg/PointCloud2
```

---

# 6. 확인 순서 — 아래에서 위로

Edge Computing Chapter 10과 같은 방식이다. 안 되는 지점에서 멈추면 그게 원인이다.

```text
① 전원·케이블        LED 가 켜지나. dmesg 에 뭐라고 나오나
       ↓
② 링크               ip addr / ip link — 랜 카드가 UP 인가
       ↓
③ IP 도달            ping -c 3 <센서 IP>
       ↓
④ 설정 채널          curl http://<IP>/... 로 응답이 오나
       ↓
⑤ 데이터 채널        tcpdump 에 UDP 패킷이 보이나       ← 경계선
       ↓
⑥ 드라이버           ros2 topic list 에 raw 토픽이 뜨나
       ↓
⑦ 변환               출력 토픽이 뜨고 hz 가 정상인가
       ↓
⑧ 내용               echo 로 값이 말이 되나 (전부 0 이 아닌가)
```

⑤가 가장 중요한 분기점이다. **여기까지 되면 남은 문제는 전부 소프트웨어**다.

---

# 7. 같은 대역에 여러 센서를 둘 때

LiDAR 두 대가 기본 IP를 그대로 쓰면 충돌한다.

```text
Velodyne VLP-16 기본   192.168.1.201
Hesai 기본             192.168.1.201     ← 같다!
```

실제로 자주 겪는 사고다. 붙이기 전에 IP를 나눠 둔다.

```text
192.168.1.201   LiDAR A
192.168.1.202   LiDAR B
192.168.1.100   호스트(Thor)
```

인터페이스가 여러 개면 라우팅도 확인한다.

```bash
ip route get 192.168.1.201     # 어느 인터페이스로 나가는지
```

DDS도 인터페이스를 골라야 한다. Edge Computing Chapter 9.5 §19의 `CYCLONEDDS_URI`가 그것이다.
**LiDAR 전용 랜으로 ROS 2 discovery가 새어 나가지 않게** 해야 한다.

---

# 8. `docs/SENSOR_CHECK.md` — 기록 형식

확인한 것은 저장소에 남긴다. 형식을 고정해 두면 비교가 쉽다.

```markdown
## Velodyne VLP-16

| 항목              | 값                                                 |
| :---------------- | :------------------------------------------------- |
| 물리 인터페이스   | Ethernet (100 Mbps)                                |
| 센서 IP           | 192.168.1.201 (기본값)                             |
| 호스트 인터페이스 | enP2p1s0                                           |
| 설정 프로토콜     | HTTP CGI                                           |
| 설정 확인         | `curl -s http://192.168.1.201/cgi/settings.json`   |
| 데이터 프로토콜   | UDP                                                |
| 데이터 포트       | 2368                                               |
| 상태 포트         | 8308                                               |
| ROS 드라이버      | `velodyne_driver` + `velodyne_pointcloud`          |
| raw 토픽          | `/velodyne_packets`                                |
| 출력 토픽         | `/velodyne_points` (`sensor_msgs/msg/PointCloud2`) |
| 기대 주기         | 10 Hz                                              |
| 확인일            | 2026-09-08                                         |
| 비고              | FOV 변경 후 저장 절차는 매뉴얼 확인 필요           |
```

**"확인일"과 "비고"가 실제로 가장 쓸모 있다.**
반년 뒤에 이 표를 볼 때, 그때 무엇이 미확인이었는지가 남아 있어야 한다.

---

# 9. Mini Practice

지금 붙어 있는 센서 하나로 ①~⑧을 끝까지 해본다.

```bash
# ②③ 링크와 도달
ip addr
ip route get 192.168.1.201
ping -c 3 192.168.1.201

# ④ 설정 채널
curl -s --max-time 3 http://192.168.1.201/cgi/settings.json | python3 -m json.tool | head -20

# ⑤ 데이터 채널  ← 경계선
sudo tcpdump -i enP2p1s0 -n -c 10 'udp port 2368'

# ⑥⑦ 드라이버와 토픽
ros2 topic list | grep -i velodyne
ros2 topic hz /velodyne_points

# ⑧ 내용
ros2 topic echo /velodyne_points --once | head -20
```

그 결과를 §8 형식으로 `docs/SENSOR_CHECK.md`에 적는다.
**모르는 칸은 비워 두지 말고 "미확인"이라고 적는다.**

---

# 10. 오늘의 핵심

```text
              센서 하나 = 두 개의 통신 + 두 단계의 토픽

   설정 채널                        데이터 채널
   HTTP / TCP / SDK / JSON          UDP / TCP / USB
   내가 물어본다                     센서가 계속 쏜다
        │                                │
        │                          tcpdump 로 확인  ← 경계선
        │                                ▼
        │                          제조사 ROS 드라이버
        │                                │
        │                          raw 토픽 (/xxx_packets)
        │                                ▼
        │                          출력 토픽 (PointCloud2)
        └────────── 둘 다 확인해야 한다 ──────────┘
```

---

# 11. 반드시 구분할 것

```text
설정 채널  ≠  데이터 채널
   ping 이 된다고 점군이 오는 게 아니다

raw 토픽  ≠  출력 토픽
   /velodyne_packets 와 /velodyne_points 는 다른 단계

토픽이 있다  ≠  데이터가 온다
   ros2 topic hz 로 실제 주기를 봐야 한다

hz 가 정상  ≠  값이 정상
   전부 0 이거나 NaN 일 수 있다. echo 로 본다

UDP  ≠  TCP
   UDP 는 놓쳐도 알려주지 않는다

센서 기본 IP
   제조사가 달라도 같을 수 있다 (다들 192.168.1.201)

tcpdump 는 호스트에서
   컨테이너 안에서는 sudo 가 막히는 경우가 많다
```

---

# 12. Chapter 연결

```text
Chapter 1  ← 여기
7가지 확인 항목과 순서

Chapter 2
Velodyne VLP-16 — 전부 채운 예시

Chapter 3
다른 LiDAR — Hesai, Livox, Ouster, BLK ARC

Chapter 4
카메라 — RealSense D435i

Edge Computing Chapter 5
Hardware Interfaces — 물리 계층

Edge Computing Chapter 8
Robot Networking — 인터페이스가 여러 개일 때

Edge Computing Chapter 10
Debugging — 계층별로 좁혀 들어가기
```
