---
title: "Chapter 4. 카메라 — RealSense D435i"
importance: 5
---

> **Goal:** USB 센서가 Ethernet 센서와 무엇이 다른지 알고,
> `lsusb` / `dmesg` / `rs-enumerate-devices`로 계층별로 확인할 수 있다.

Chapter 2·3의 LiDAR는 전부 Ethernet이었다. 카메라는 대개 USB다.
**확인 항목은 같지만 확인 방법이 전부 바뀐다.**

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `lsusb` · `dmesg -w` · `rs-enumerate-devices`

| 명령어                                         | 하는 일                                |
| :--------------------------------------------- | :------------------------------------- |
| `lsusb`                                        | 장치가 인식됐나                        |
| `lsusb -t`                                     | **어느 USB 버스에 어떤 속도로 붙었나** |
| `dmesg -w`                                     | 꽂는 순간 커널 메시지 (켜 놓고 꽂는다) |
| `ls -l /dev/video*`                            | video device node와 권한               |
| `rs-enumerate-devices`                         | librealsense가 장치를 보나             |
| `rs-enumerate-devices -s`                      | 짧게 (모델·시리얼·펌웨어)              |
| `realsense-viewer`                             | GUI로 직접 보기                        |
| `ros2 launch realsense2_camera rs_launch.py`   | 드라이버 실행                          |
| `ros2 topic hz /camera/camera/color/image_raw` | 실제 fps                               |
| `v4l2-ctl --list-devices`                      | V4L2 수준에서 보이나                   |

---

# 1. 한 장 요약

| 항목            | 값                                                                                                        |
| :-------------- | :-------------------------------------------------------------------------------------------------------- |
| 물리 인터페이스 | **USB 3.x** (Type-C)                                                                                      |
| 설정 프로토콜   | `librealsense` SDK / ROS 파라미터                                                                         |
| 데이터 프로토콜 | USB stream (UVC 기반)                                                                                     |
| 주소·포트       | 없음. **시리얼 번호**로 구분                                                                              |
| ROS 드라이버    | `realsense2_camera` ([IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros))     |
| 출력 토픽       | `/camera/camera/color/image_raw`<br>`/camera/camera/depth/image_rect_raw`<br>`/camera/camera/imu` (D435i) |
| 내장 IMU        | **있음** (D435i의 `i`가 그 뜻)                                                                            |

```text
Ethernet 센서               USB 센서
──────────────────         ────────────────────────
IP 주소로 찾는다             시리얼 번호로 찾는다
포트가 있다                  포트가 없다
ping / tcpdump              lsusb / dmesg
방화벽이 막을 수 있다         전원·케이블·대역폭이 문제
여러 대 = IP 충돌            여러 대 = USB 대역폭 경쟁
```

---

# 2. USB에서 가장 많이 터지는 것

Ethernet 센서와 달리, USB는 **케이블과 전원이 절반**이다.

```text
① 케이블      USB 2.0 케이블을 쓰면 3.0 속도가 안 난다. 겉으로는 똑같이 생겼다
② 전원        Jetson 의 USB 포트가 전류를 못 대주면 간헐적으로 끊긴다
③ 대역폭      한 컨트롤러에 카메라 두 대를 물리면 둘 다 프레임이 떨어진다
④ 권한        /dev/video* 접근 권한. udev rule 이 필요할 수 있다
```

②③은 **간헐적으로 실패**해서 특히 골치 아프다. 로그를 봐야 보인다.

속도 확인이 첫 번째다.

```bash
lsusb -t
```

```text
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/4p, 5000M
    |__ Port 2: Dev 3, If 0, Class=Video, Driver=uvcvideo, 5000M
                                                            ^^^^^
                                              5000M = USB 3.0 으로 붙었다
                                              480M  = USB 2.0. 케이블 의심
```

**`480M`이 보이면 케이블이나 포트 문제다.** D435i는 USB 3.0이 아니면 해상도·fps가 제한된다.

---

# 3. 계층별 확인 — 아래에서 위로

Ethernet의 `ping → tcpdump → 토픽` 순서에 대응하는 USB 버전이다.

```text
① 커널이 보나        dmesg -w  를 켜 놓고 꽂는다
       ↓
② USB 목록에 있나    lsusb | grep -i intel
       ↓
③ 속도가 맞나        lsusb -t   → 5000M 인가
       ↓
④ V4L2 로 보이나     v4l2-ctl --list-devices,  ls -l /dev/video*
       ↓
⑤ SDK 가 보나       rs-enumerate-devices        ← 경계선
       ↓
⑥ 드라이버          ros2 launch realsense2_camera rs_launch.py
       ↓
⑦ 토픽              ros2 topic hz /camera/camera/color/image_raw
       ↓
⑧ 내용              rviz2 로 보거나 echo
```

⑤가 Chapter 1 §6의 `tcpdump`에 해당하는 경계선이다.
**`rs-enumerate-devices`가 장치를 보면 하드웨어·드라이버 문제는 끝**이고,
그 위는 전부 ROS 설정 문제다.

```bash
rs-enumerate-devices -s
```

```text
Device Name        Serial Number    Firmware Version
Intel RealSense D435I   012345678901   5.13.0.50
```

**펌웨어 버전을 기록해 두는 게 중요하다.** librealsense 버전과 궁합이 있다.

---

# 4. 토픽 구조

```bash
ros2 launch realsense2_camera rs_launch.py
ros2 topic list
```

```text
/camera/camera/color/camera_info
/camera/camera/color/image_raw
/camera/camera/color/metadata
/camera/camera/depth/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/depth/metadata
/camera/camera/extrinsics/depth_to_color
```

이름이 `/camera/camera/...`로 두 번 겹치는 이유는 **네임스페이스와 노드 이름이 둘 다 기본값 `camera`**이기 때문이다. 바꿀 수 있다.

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=front camera_name:=d435i
# → /front/d435i/color/image_raw
```

카메라 두 대를 붙일 때 **반드시 나눠야 한다.**

IMU는 기본으로 안 켜질 수 있다.

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true enable_accel:=true unite_imu_method:=2
```

```text
unite_imu_method
  0   합치지 않음 → /gyro/sample, /accel/sample 로 따로
  1   copy
  2   linear_interpolation     ← 하나의 /imu 토픽으로 합쳐진다
```

**LIO에 쓰려면 합쳐진 `sensor_msgs/msg/Imu` 하나가 필요**하므로 대개 `2`를 쓴다.

`extrinsics/depth_to_color`도 눈여겨볼 것이다.
depth와 color가 물리적으로 떨어져 있으므로, 정렬하려면 이 변환이 필요하다.

---

# 5. Docker 안에서

USB 센서는 Ethernet보다 컨테이너 설정이 까다롭다.

```bash
docker run -it \
  --network host \
  --device /dev/video0 \
  --device /dev/video1 \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  my-ros-image bash
```

```text
--device /dev/videoN     그 노드만 넘긴다 (권장)
-v /dev:/dev             전부 넘긴다. 재연결 시 노드 번호가 바뀌어도 대응됨
-v /run/udev:/run/udev   librealsense 가 udev 정보를 읽는다
--privileged             마지막 수단. 가급적 피한다
```

**카메라를 뽑았다 꽂으면 `/dev/videoN` 번호가 바뀐다.**
`--device`로 특정 노드만 넘겼다면 컨테이너가 그걸 잃는다.
개발 중에는 `-v /dev:/dev`가 편하고, 운영에서는 udev rule로 고정 이름을 만드는 편이 낫다.

호스트에 udev rule이 깔려 있어야 한다는 점도 주의한다.

```bash
# 호스트에서
ls /etc/udev/rules.d/ | grep -i realsense
```

(Edge Computing Chapter 9.5, Chapter 14)

---

# 6. 안 될 때

```text
증상                              먼저 볼 것
────────────────────────────      ──────────────────────────────────
lsusb 에 아예 없다                 케이블, 전원, 포트
                                  dmesg -w 켜고 다시 꽂기

lsusb -t 가 480M                   USB 2.0 케이블/포트
                                  D435i 는 3.0 이어야 제 성능

rs-enumerate-devices 가 빈 목록     권한 (udev rule)
                                  librealsense 설치·버전
                                  펌웨어 호환

토픽은 뜨는데 fps 가 낮다          USB 대역폭 경쟁 (다른 USB 장치)
                                  해상도·fps 설정
                                  CPU 부하

간헐적으로 끊긴다                  전원 부족이 1순위
                                  dmesg 에 disconnect 가 찍히나
                                  셀프파워 허브 시도

/dev/videoN 번호가 계속 바뀐다      udev rule 로 고정 이름 만들기
```

**"간헐적"이면 거의 항상 전원 아니면 케이블이다.**

---

# 7. 카메라는 저장 용량을 가장 많이 먹는다

Edge Computing Chapter 13의 계산을 다시 보면 카메라가 압도적이다.

```text
                                    MB/s     GB/시간
VLP-16 LiDAR                          4.6       16.6
1080p30 카메라, JPEG 약 10:1         18.7       67.2
1080p30 카메라, 무압축 RGB8         186.6      671.8   ← LiDAR 의 40배
```

D435i는 color + depth + IMU를 동시에 낸다.
**`ros2 bag record -a`를 무심코 돌리면 몇 분 만에 디스크가 찬다.**

```bash
# 필요한 것만
ros2 bag record /camera/camera/color/image_raw/compressed /camera/camera/imu
```

---

# 8. 기록

```markdown
## Intel RealSense D435i

| 항목            | 값                                                           |
| :-------------- | :----------------------------------------------------------- |
| 물리 인터페이스 | USB 3.x (Type-C)                                             |
| 식별            | 시리얼 번호                                                  |
| 설정 프로토콜   | librealsense SDK / ROS 파라미터                              |
| 데이터 프로토콜 | USB stream (UVC)                                             |
| ROS 드라이버    | realsense2_camera                                            |
| 출력 토픽       | /camera/camera/color/image_raw                               |
|                 | /camera/camera/depth/image_rect_raw                          |
|                 | /camera/camera/imu (enable_gyro/accel + unite_imu_method:=2) |
| 내장 IMU        | 있음                                                         |
| 펌웨어          | (rs-enumerate-devices -s 로 확인해 기록)                     |
| librealsense    | (버전 기록)                                                  |
| 확인일          | 2026-09-08                                                   |
| 비고            | lsusb -t 가 5000M 인지 매번 확인                             |
```

---

# 9. Chapter 연결

```text
Chapter 1   7가지 확인 항목
Chapter 2   VLP-16 — Ethernet + HTTP + UDP
Chapter 3   다른 LiDAR
Chapter 4   ← 여기. USB 센서는 확인 방법이 다르다

Edge Computing Chapter 5    USB, device node, udev
Edge Computing Chapter 9.5  --device, /run/udev 마운트
Edge Computing Chapter 13   카메라가 저장을 가장 많이 먹는다
Edge Computing Chapter 14   udev rule 로 이름 고정하기
```

**출처:** 토픽 이름과 파라미터는
[IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) README에서 확인했다.
펌웨어·librealsense 버전 궁합은 장비에서 직접 확인해 기록해야 한다.
