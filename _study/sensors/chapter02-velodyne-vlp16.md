---
title: "Chapter 2. Velodyne VLP-16"
importance: 3
---

> **Goal:** VLP-16의 설정 채널(HTTP)과 데이터 채널(UDP)을 구분해서 다루고,
> `curl`과 `tcpdump`로 각각을 직접 확인할 수 있다.

Chapter 1의 7가지를 실제로 전부 채운 첫 예시다.
다른 센서는 이 표의 값만 바뀐다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `curl .../settings.json` · `tcpdump udp port 2368`

| 명령어                                                                       | 하는 일            |
| :--------------------------------------------------------------------------- | :----------------- |
| `ping -c 3 192.168.1.201`                                                    | 센서에 닿는지      |
| `curl -s http://192.168.1.201/cgi/settings.json \| python3 -m json.tool`     | **현재 설정 전체** |
| `curl -s http://192.168.1.201/cgi/info.json`                                 | 모델·시리얼·펌웨어 |
| `curl -s http://192.168.1.201/cgi/status.json`                               | 모터 상태·GPS      |
| `curl -s http://192.168.1.201/cgi/diag.json`                                 | 진단 값            |
| `curl -X POST http://192.168.1.201/cgi/setting/fov --data 'start=0&end=359'` | FOV 변경           |
| `curl -X POST http://192.168.1.201/cgi/setting --data 'rpm=600'`             | 회전 속도          |
| `sudo tcpdump -i enP2p1s0 -n -c 10 'udp port 2368'`                          | **점군 패킷 확인** |
| `sudo tcpdump -i enP2p1s0 -n 'udp port 8308'`                                | 위치·상태 패킷     |
| `ros2 topic hz /velodyne_points`                                             | 실제 주기 (10 Hz)  |

---

# 1. 한 장 요약

| 항목            | 값                                                 |
| :-------------- | :------------------------------------------------- |
| 물리 인터페이스 | Ethernet (100 Mbps)                                |
| 센서 기본 IP    | `192.168.1.201`                                    |
| 설정 프로토콜   | **HTTP CGI** (내장 웹서버)                         |
| 데이터 프로토콜 | **UDP**                                            |
| 점군 포트       | `2368`                                             |
| 위치·상태 포트  | `8308`                                             |
| ROS 드라이버    | `velodyne_driver` + `velodyne_pointcloud`          |
| raw 토픽        | `/velodyne_packets`                                |
| 출력 토픽       | `/velodyne_points` (`sensor_msgs/msg/PointCloud2`) |
| 기대 주기       | 10 Hz (RPM 600 기준)                               |

```text
    Thor  ──── HTTP ────────▶  VLP-16      설정·조회
    Thor  ◀─── UDP 2368 ─────  VLP-16      점군
    Thor  ◀─── UDP 8308 ─────  VLP-16      위치·상태
```

---

# 2. 설정 채널 — HTTP CGI

VLP-16 안에는 작은 웹서버가 들어 있다. 브라우저로 `http://192.168.1.201`을 열면 UI가 나오고,
같은 것을 `curl`로도 할 수 있다.

읽기는 GET이다.

```bash
curl -s http://192.168.1.201/cgi/settings.json | python3 -m json.tool
```

쓰기는 POST다.

```bash
curl -X POST \
  http://192.168.1.201/cgi/setting/fov \
  --data 'start=0&end=359'
```

**바꿨으면 반드시 다시 읽어서 확인한다.**

```bash
curl -s http://192.168.1.201/cgi/settings.json | python3 -m json.tool
```

그리고 전원을 껐다 켰을 때 유지되는지는 별개 문제다.
**영구 반영에는 저장 절차가 따로 필요할 수 있으므로 매뉴얼을 확인해야 한다.**
(이 페이지에서 검증하지 않은 항목이다.)

FOV를 좁히면 데이터량이 준다. Edge Computing Chapter 13의 저장 계산과 직접 연결된다.

```text
FOV 360°  →  16 × 1800 × 10Hz = 288,000 점/초  ≈ 4.6 MB/s
FOV 180°  →  절반
```

---

# 3. 데이터 채널 — UDP

점군은 HTTP로 오지 않는다. **UDP 2368로 센서가 일방적으로 쏜다.**

```bash
sudo tcpdump -i enP2p1s0 -n -c 10 'udp port 2368'
```

```text
패킷이 보인다      → 센서가 내 쪽으로 쏘고 있다. 여기까지가 네트워크
아무것도 없다      → 아래 §6 으로
```

`8308`은 위치·상태 계열이다. GPS/PPS를 붙였다면 여기로 온다.

```bash
sudo tcpdump -i enP2p1s0 -n 'udp port 8308'
```

**`tcpdump`는 호스트에서 돌린다.** 컨테이너 안에서는 이렇게 막히는 경우가 많다.

```text
$ docker exec sd-slam bash -c 'sudo -n timeout 10 tcpdump ...'
sudo: a password is required
```

컨테이너 안에서 꼭 해야 한다면 `--privileged`나 `--cap-add=NET_ADMIN,NET_RAW`가 필요하고,
`tcpdump`도 이미지에 들어 있어야 한다. (Edge Computing Chapter 9.5)
**진단은 호스트에서 하는 편이 빠르다.**

---

# 4. ROS 드라이버 — 두 단계

Velodyne 드라이버는 **패킷 수신**과 **점군 변환**이 분리되어 있다.

```text
UDP 2368
    ▼
velodyne_driver        패킷을 받아 그대로 ROS 메시지로
    ▼
/velodyne_packets      velodyne_msgs/msg/VelodyneScan   ← raw
    ▼
velodyne_pointcloud    캘리브레이션을 적용해 3D 점으로 변환
    ▼
/velodyne_points       sensor_msgs/msg/PointCloud2      ← 실제로 쓰는 것
```

이 분리 덕분에 진단이 정확해진다.

```text
/velodyne_packets 만 있다    네트워크 OK, 변환 노드가 안 떴거나 캘리브 파일 문제
둘 다 없다                   드라이버가 UDP 를 못 받고 있다
둘 다 있는데 hz 가 낮다      패킷 손실
```

```bash
ros2 topic list | grep velodyne
ros2 topic hz   /velodyne_packets
ros2 topic hz   /velodyne_points
ros2 topic type /velodyne_points
```

`10 Hz`가 나와야 정상이다 (RPM 600 기준).

```text
RPM 300   → 5 Hz
RPM 600   → 10 Hz    기본값
RPM 1200  → 20 Hz    한 바퀴당 점이 줄어든다
```

---

# 5. 확인 순서 전체

```bash
# ① 링크
ip addr show enP2p1s0
ip route get 192.168.1.201

# ② 도달
ping -c 3 192.168.1.201

# ③ 설정 채널
curl -s --max-time 3 http://192.168.1.201/cgi/info.json | python3 -m json.tool

# ④ 데이터 채널  ← 경계선
sudo tcpdump -i enP2p1s0 -n -c 10 'udp port 2368'

# ⑤ 드라이버
ros2 topic hz /velodyne_packets

# ⑥ 변환
ros2 topic hz /velodyne_points

# ⑦ 내용
ros2 topic echo /velodyne_points --once | head -20
```

---

# 6. 안 될 때

```text
증상                              먼저 볼 것
────────────────────────────      ──────────────────────────────────
ping 이 안 된다                    호스트 IP 가 같은 대역인가 (192.168.1.x)
                                  ip route get 192.168.1.201
                                  케이블·링크 LED

ping 은 되는데 tcpdump 가 조용     센서의 UDP 목적지가 내가 아닐 수 있다
                                  settings.json 의 목적지 IP 확인
                                  방화벽: sudo ufw status

tcpdump 는 보이는데 토픽이 없다     드라이버의 device_ip / port 설정
                                  드라이버가 다른 인터페이스를 보고 있나

토픽은 있는데 hz 가 낮다           패킷 손실.
                                  ip -s link 로 drop 확인
                                  MTU, 스위치, CPU 부하

점이 전부 0 이거나 이상하다        캘리브레이션 파일(모델별 yaml)이 맞나
                                  VLP-16 용을 쓰고 있나
```

`ping`이 되는데 UDP가 없는 경우가 가장 흔하다.
**센서는 "내가 쏠 목적지"를 자기 설정에 들고 있다.** 그 값이 내 IP가 아니면 조용하다.

---

# 7. 주의할 점

```text
· 설정 변경은 실제 장비 동작을 바꾼다. FOV 를 좁히면 그만큼 안 보인다
· 저장 절차 없이 껐다 켜면 되돌아갈 수 있다 (매뉴얼 확인)
· 기본 IP 192.168.1.201 은 다른 제조사와 겹친다 (Chapter 3)
· 100 Mbps 링크다. 스위치가 그 이하로 협상하면 손실이 난다
      ethtool enP2p1s0 로 속도 확인
```

---

# 8. 기록

```markdown
## Velodyne VLP-16

| 항목              | 값                                               |
| :---------------- | :----------------------------------------------- |
| 물리 인터페이스   | Ethernet 100 Mbps                                |
| 센서 IP           | 192.168.1.201                                    |
| 호스트 인터페이스 | enP2p1s0                                         |
| 설정 프로토콜     | HTTP CGI                                         |
| 설정 확인         | `curl -s http://192.168.1.201/cgi/settings.json` |
| 데이터 프로토콜   | UDP                                              |
| 데이터 포트       | 2368                                             |
| 상태 포트         | 8308                                             |
| ROS 드라이버      | velodyne_driver + velodyne_pointcloud            |
| raw 토픽          | /velodyne_packets                                |
| 출력 토픽         | /velodyne_points (PointCloud2)                   |
| 기대 주기         | 10 Hz                                            |
| 확인일            | 2026-09-08                                       |
| 미확인            | FOV 변경 후 영구 저장 절차                       |
```

---

# 9. Chapter 연결

```text
Chapter 1
7가지 확인 항목

Chapter 2  ← 여기
VLP-16 — HTTP 설정 + UDP 2368

Chapter 3
Hesai / Livox / Ouster / BLK ARC — 같은 자리에 다른 값이 들어간다

Edge Computing Chapter 13
FOV·RPM 이 저장 용량으로 이어지는 계산
```

**참고:** VLP-16 User Manual의 웹 인터페이스·CGI API 절.
이 페이지의 포트와 토픽은 실제 확인한 값이고, 저장 절차는 매뉴얼 확인이 남아 있다.
