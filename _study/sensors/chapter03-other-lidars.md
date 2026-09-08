---
title: "Chapter 3. Hesai · Livox · Ouster · BLK ARC"
importance: 4
---

> **Goal:** 같은 7가지 항목에 제조사마다 어떤 값이 들어가는지 비교하고,
> 처음 보는 LiDAR도 같은 순서로 확인할 수 있다.

Chapter 2의 VLP-16은 **HTTP 설정 + UDP 2368**이었다.
다른 제조사도 큰 구조는 같지만 **채워지는 값이 전부 다르다.**

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `tcpdump 'host <센서IP>'` · `ros2 topic hz`

| 명령어                                         | 하는 일                                  |
| :--------------------------------------------- | :--------------------------------------- |
| `sudo tcpdump -i eth1 -n 'host 192.168.1.201'` | **포트를 모를 때** 그 센서와 오가는 전부 |
| `sudo tcpdump -i eth1 -n 'udp' -c 20`          | 어느 포트로 오는지 훑기                  |
| `nmap -sU -p 1-10000 192.168.1.201`            | UDP 포트 스캔 (느림, 최후 수단)          |
| `ros2 topic list`                              | 드라이버가 낸 토픽                       |
| `ros2 topic hz <topic>`                        | 실제 주기                                |
| `cat config/*.json`                            | Livox 설정 파일                          |

---

# 1. 한 장 비교

|              | Velodyne VLP-16     | Hesai (XT/Pandar)    | Livox (Mid-360, HAP) | Ouster (OS0/1/2)                      | Leica BLK ARC |
| :----------- | :------------------ | :------------------- | :------------------- | :------------------------------------ | :------------ |
| 물리         | Ethernet            | Ethernet             | Ethernet             | Ethernet                              | **미확인**    |
| 설정         | HTTP CGI            | **PTC** (TCP 9347)   | **JSON 설정 파일**   | HTTP API                              | 전용 SDK      |
| 데이터       | UDP 2368            | UDP 2368             | UDP 57000            | UDP                                   | 전용 SDK      |
| IMU          | 없음                | 모델별               | UDP 58000            | 있음                                  | 내장          |
| 기본 IP      | 192.168.1.201       | **192.168.1.201**    | 설정 파일에 명시     | DHCP/link-local + `os-<시리얼>.local` | 미확인        |
| ROS 드라이버 | `velodyne_driver`   | `HesaiLidar_ROS_2.0` | `livox_ros_driver2`  | `ouster-ros`                          | 미확인        |
| raw 토픽     | `/velodyne_packets` | `/lidar_packets`     | —                    | —                                     | —             |
| 출력 토픽    | `/velodyne_points`  | `/lidar_points`      | `/livox/lidar`       | `/ouster/points`                      | 미확인        |

**Velodyne과 Hesai의 기본 IP가 같다.** 둘을 같은 대역에 붙이면 바로 충돌한다.

---

# 2. Hesai — 설정이 HTTP가 아니다

Velodyne을 먼저 만진 사람이 가장 헷갈리는 지점이다.
Hesai는 **PTC(Pandar Terminal Control)**라는 자체 TCP 프로토콜로 설정한다.

```text
설정    TCP 9347   PTC (SSL 옵션 있음)     ← curl 로 안 된다
데이터  UDP 2368   점군
```

드라이버 설정 파일에 그대로 드러난다.

```yaml
lidar_udp_type:
  device_ip_address: 192.168.1.201 # 센서 IP
  udp_port: 2368 # 점군 UDP 포트
  ptc_port: 9347 # PTC 포트
  use_ptc_connected: true # PTC 를 안 쓸 거면 false
  ptc_mode: 0 # 0=tcp, 1=tcp_ssl
ros_send_point_cloud_topic: /lidar_points
ros_frame_id: hesai_lidar
```

```text
· 점군만 받고 싶으면 use_ptc_connected: false 로 두고 UDP 만 받아도 된다
· 빌드 시 PTC SSL 을 끄려면 -DWITH_PTCS_USE=OFF
```

드라이버는 [HesaiTechnology/HesaiLidar_ROS_2.0](https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0)이고
내부적으로 `HesaiLidar_SDK_2.0`을 쓴다. UDP 패킷을 받아 파싱해서 `/lidar_points`로 낸다.

**VLP-16의 `curl` 습관이 여기서는 통하지 않는다.** SDK나 제조사 도구를 써야 한다.

---

# 3. Livox — 설정이 파일이다

Livox는 또 다르다. 센서에 물어보는 게 아니라 **드라이버가 읽는 JSON 파일에 다 적는다.**

```json
{
  "lidar_configs": [
    {
      "ip": "192.168.1.12",
      "cmd_data_port": 56000,
      "push_msg_port": 0,
      "point_data_port": 57000,
      "imu_data_port": 58000,
      "log_data_port": 59000
    }
  ]
}
```

포트가 **용도별로 넷**이라는 게 특징이다.

```text
56000   명령 (설정)
57000   점군
58000   IMU        ← LiDAR 안에 IMU 가 들어 있다
59000   로그
```

launch 파일이 이 JSON 경로를 `user_config_path`로 가리킨다.

```bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

출력 형식을 파라미터로 고를 수 있다.

```text
xfer_format = 0    Livox PointCloud2 (PointXYZRTLT)   기본값
              1    Livox CustomMsg
              2    표준 PointCloud2 (pcl::PointXYZI)
```

**`xfer_format`을 확인하지 않으면 다른 노드가 메시지를 못 읽는다.**
FAST-LIO2 계열은 `CustomMsg`를 기대하는 경우가 많고,
일반 PCL 도구는 표준 `PointCloud2`가 필요하다.

드라이버는 [Livox-SDK/livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)이고
그 아래에 `Livox-SDK2`가 따로 설치되어 있어야 한다.
Edge Computing Chapter 9.5의 Dockerfile에서 `/usr/local/lib`에 굽던 것이 이것이다.

---

# 4. Ouster — IP가 고정이 아니다

Ouster는 기본 IP를 박아 두지 않는다.

```text
DHCP 가 있으면      DHCP 로 받는다
없으면              link-local (169.254.0.0/16) 로 자기가 정한다
이름                os-<시리얼번호>.local   ← mDNS 로 찾는다
```

**IP를 모르는 상태에서 시작한다**는 게 다른 점이다.

```bash
ping os-991234567890.local          # 시리얼로 찾는다
avahi-browse -a | grep -i ouster    # mDNS 로 훑기
```

설정은 HTTP API를 쓰고, mDNS 안내에 TCP `7501`이 함께 뜬다.

드라이버는 [ouster-lidar/ouster-ros](https://github.com/ouster-lidar/ouster-ros)이고
ROS 2는 `ros2` 브랜치를 써야 한다 (`master`는 ROS 1).

```text
/ouster/points     점군
/ouster/imu        내장 IMU
/ouster/points2    dual return 을 켰을 때 두 번째 점군
```

Ouster도 **IMU가 내장**이라 별도 IMU 없이 LIO를 돌릴 수 있다.

---

# 5. Leica BLK ARC — 여기서 멈춘다

BLK ARC는 위 넷과 성격이 다르다. 스캐너 자체가 SLAM을 돌리는 통합 장비에 가깝다.

**이 페이지에서는 포트나 프로토콜을 확인하지 못했다.**
공개 문서로 검증되지 않은 값을 적으면 나중에 그게 사실인 줄 알고 쓰게 되므로,
확인 전까지는 빈칸으로 둔다.

```markdown
## Leica BLK ARC

| 항목            | 값                                   |
| :-------------- | :----------------------------------- |
| 물리 인터페이스 | 미확인                               |
| 설정 프로토콜   | 미확인 (전용 SDK/API 로 알려져 있음) |
| 데이터 프로토콜 | 미확인                               |
| 포트            | 미확인                               |
| ROS 드라이버    | 미확인                               |
| 출력 토픽       | 미확인                               |
| 확인일          | —                                    |
| 다음 할 일      | 장비 매뉴얼 / 제조사 SDK 문서 확인   |
```

장비를 실제로 받으면 Chapter 1 §9의 순서를 그대로 돌려서 채우면 된다.
**포트를 모를 때는 이렇게 훑는다.**

```bash
# 그 호스트와 오가는 모든 패킷
sudo tcpdump -i eth1 -n 'host <센서IP>' -c 50

# 어떤 UDP 포트가 쓰이는지 통계
sudo tcpdump -i eth1 -n udp -c 200 | awk '{print $5}' | sort | uniq -c | sort -rn | head
```

---

# 6. 처음 보는 LiDAR를 받았을 때

```text
① 매뉴얼에서 기본 IP 를 찾는다        못 찾으면 ②
② 랜선만 꽂고 tcpdump 로 훑는다       센서가 뭔가 쏘고 있으면 IP 가 보인다
       sudo tcpdump -i eth1 -n -c 20
③ 그 IP 대역에 맞춰 호스트 IP 를 잡는다
④ ping → 설정 채널(HTTP? TCP? SDK?) 확인
⑤ tcpdump 로 데이터 포트를 특정한다
⑥ 제조사 ROS 드라이버를 찾아 설정에 ④⑤ 값을 넣는다
⑦ ros2 topic hz 로 주기를 확인한다
⑧ SENSOR_CHECK.md 에 기록한다
```

②가 의외로 잘 먹힌다. 많은 LiDAR가 **아무 설정 없이도 켜지면 뭔가를 쏜다.**

---

# 7. 여러 대를 같이 붙일 때

```text
문제
  Velodyne 기본  192.168.1.201
  Hesai 기본     192.168.1.201    ← 충돌

해결
  각각 다른 IP 로 바꾼다
  또는 물리적으로 다른 랜 카드에 분리한다
```

인터페이스를 나눴다면 DDS도 골라줘야 한다.

```bash
ip route get 192.168.1.201     # 어느 인터페이스로 나가나
```

`CYCLONEDDS_URI`로 ROS 2가 쓸 인터페이스를 고정한다.
LiDAR 전용 랜으로 discovery가 새면 대역폭을 잡아먹는다.
(Edge Computing Chapter 9.5 §19)

---

# 8. 오늘의 핵심

```text
            같은 7칸, 다른 값

              설정            데이터           IMU
   ─────────  ─────────────   ─────────────   ──────
   Velodyne   HTTP CGI        UDP 2368        없음
   Hesai      TCP 9347 PTC    UDP 2368        모델별
   Livox      JSON 파일       UDP 57000       UDP 58000
   Ouster     HTTP API        UDP             내장
   BLK ARC    전용 SDK        미확인          내장

   "curl 로 설정한다" 는 Velodyne 의 방식일 뿐이다
```

---

# 9. 반드시 구분할 것

```text
HTTP 설정  ≠  모든 LiDAR 의 방식
   Hesai 는 PTC(TCP), Livox 는 JSON 파일

UDP 2368  ≠  표준 포트
   Velodyne 과 Hesai 가 같을 뿐. Livox 는 57000

기본 IP 가 다르다  ≠  사실
   Velodyne 과 Hesai 둘 다 192.168.1.201

Livox xfer_format
   0/1/2 중 무엇인지에 따라 받는 쪽이 못 읽는다

Ouster 는 IP 가 고정이 아니다
   os-<시리얼>.local 로 찾는다

ouster-ros 의 master 브랜치
   ROS 1 이다. ROS 2 는 ros2 브랜치

미확인  ≠  없음
   확인 못 한 것은 "미확인"이라고 적는다
```

---

# 10. Chapter 연결

```text
Chapter 1   7가지 확인 항목
Chapter 2   VLP-16 — 전부 채운 예시
Chapter 3   ← 여기. 같은 칸에 다른 값
Chapter 4   카메라 — RealSense D435i

Edge Computing Chapter 9.5   CYCLONEDDS_URI, Livox-SDK2 를 이미지에 굽기
Edge Computing Chapter 13    센서별 데이터량 계산
```

**출처:** 각 드라이버 저장소의 README와 설정 파일에서 확인했다 —
[HesaiLidar_ROS_2.0](https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0),
[livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2),
[ouster-ros](https://github.com/ouster-lidar/ouster-ros).
BLK ARC는 확인하지 못해 빈칸으로 두었다.
