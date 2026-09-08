---
title: "Chapter 9.5. Docker 실전 — 컨테이너 파일시스템과 개발 워크플로우"
importance: 10.5
---

> **Goal:** 컨테이너 안에서 보이는 경로 중 **어떤 것이 이미지에 구워진 것이고 어떤 것이 호스트에서 빌려온 것인지** 구분하고,
> `build.sh` / `run.sh` / `colcon build`로 이어지는 하루 작업 흐름을 명령어 단위로 실행할 수 있다.
> 또한 `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, X11 같은 설정이
> **이미지·컨테이너 중 어디에 박히는지**를 구분하고, 컨테이너 안 ROS 2가 상대 로봇과
> 통신되는지 순서대로 확인할 수 있다.

Chapter 9는 Docker가 무엇인지, Jetson에서 왜 쓰는지를 다뤘다.
이 chapter는 그 다음 단계다. 개념을 알아도 막상 컨테이너에 들어가면
`/opt/ros/humble`과 `/ws`가 **왜 서로 다르게 동작하는지** 헷갈리는 경우가 많다.
그 구분 하나만 정확히 잡으면 나머지 워크플로우는 자동으로 따라온다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `./docker/run.sh` · `docker exec -it sd-slam bash` · `colcon build`

| 명령어                                    | 하는 일                                             |
| :---------------------------------------- | :-------------------------------------------------- |
| `./docker/build.sh`                       | 이미지 생성. **Dockerfile을 고쳤을 때만** (10~20분) |
| `./docker/run.sh`                         | 컨테이너 생성 or 시작 + 접속 (매번, 1초)            |
| `docker exec -it sd-slam bash`            | 터미널 하나 더 열기                                 |
| `docker ps`                               | 컨테이너가 살아 있는지                              |
| `docker stop sd-slam`                     | 잠시 끄기 (설치한 것은 남는다)                      |
| `docker rm -f sd-slam`                    | 삭제. **`-e` 값을 바꾸려면 이게 필요**              |
| `ls /opt/ros/humble`                      | 이미지에만 있는 것                                  |
| `ls /ws`                                  | 호스트에서 빌려온 것                                |
| `colcon build --symlink-install`          | overlay 빌드 (결과는 호스트 `/ws/install`에 남음)   |
| `source install/setup.bash`               | `docker exec`로 새로 들어갈 때마다 다시 필요        |
| `env \| grep -E 'ROS_\|CYCLONE\|DISPLAY'` | 환경변수가 전달됐는지                               |
| `echo $ROS_DOMAIN_ID`                     | 토픽이 안 보일 때 1순위                             |
| `echo $RMW_IMPLEMENTATION`                | 어느 DDS를 쓰는지                                   |
| `xhost +local:`                           | GUI를 띄우기 전 **호스트에서** 한 번                |
| `rqt_graph`                               | node·topic 연결 방향 보기                           |
| `sudo chown -R $(id -u):$(id -g) .`       | root 소유가 된 파일 되돌리기                        |

---

# 1. 컨테이너는 자기만의 완전한 파일시스템을 가진다

가장 먼저 버려야 할 착각이 있다.

```text
"컨테이너는 호스트 위에서 도는 프로그램이니까
 호스트 파일을 그냥 보는 거겠지"
```

아니다.

컨테이너에 들어가서 `ls /`를 하면 보이는 것은 **이미지가 통째로 제공한 파일시스템**이다.
호스트의 `/`가 아니다.

그래서 이런 일이 가능하다.

```text
호스트    : Ubuntu 24.04 (noble)
컨테이너  : Ubuntu 22.04 (jammy)
```

ROS 2 Humble은 jammy용으로 배포된다.
호스트를 24.04로 쓰면서도 컨테이너 안에서 Humble을 쓸 수 있는 이유가 이것이다.
커널만 공유하고, **그 위의 파일시스템은 완전히 별개**다.

---

# 2. `/opt`와 `/ws`의 실체

실제 SLAM 개발 환경을 예로 보자.

```text
호스트 (Ubuntu 24.04 noble)              컨테이너 (Ubuntu 22.04 jammy)
────────────────────────────             ──────────────────────────────
                                         /                ← 이미지가 통째로 제공
/opt/ros   ✗ 없음                        /opt/ros/humble/ ← 이미지 안에만 존재
                                            rclcpp, tf2, velodyne_driver,
                                            realsense2_camera, ...
                                         /usr/local/cargo/ ← Rust, 이미지 안에만
                                         /usr/local/lib/   ← Livox-SDK2, 이미지 안에만

~/local_ws/SeoulDynamics.SLAM/  ◄══════► /ws/          ← 같은 파일! (bind mount)
   ├── ros2/                    ◄══════►    ├── ros2/
   ├── slam/                    ◄══════►    ├── slam/
   ├── config/                  ◄══════►    ├── config/
   └── scripts/                 ◄══════►    └── scripts/

~/local_ws/Vision60.Autonomy/            (컨테이너에서 안 보임 — 마운트 안 했으니)
~/anaconda3/                             (안 보임)
```

여기서 중요한 것은 **마운트하지 않은 것은 안 보인다**는 점이다.
호스트에 있어도, 같은 홈 디렉토리 안에 있어도, `-v`로 연결하지 않았으면 컨테이너는 그 존재를 모른다.

---

# 3. 두 종류의 차이

|               | `/opt/ros/humble/`                                          | `/ws/`                                        |
| ------------- | ----------------------------------------------------------- | --------------------------------------------- |
| 어디서 왔나   | 이미지에 구워짐 — Dockerfile의 `RUN apt-get install`이 만듦 | 호스트에서 빌려옴 — `run.sh`의 `-v "$WS":/ws` |
| 호스트에 있나 | ✗ 없음                                                      | ✓ 있음. 같은 파일                             |
| 누가 만드나   | `build.sh` (10~20분, 1회)                                   | 없음. 그냥 연결만                             |
| 컨테이너 끄면 | 이미지에 남음 (재사용)                                      | 호스트에 그대로 (애초에 호스트 파일)          |
| 고치면        | 이미지 다시 빌드해야 함                                     | 즉시 반영. 양방향                             |

이 표가 이 chapter의 전부라고 해도 된다.
아래 내용은 전부 이 표에서 파생된다.

---

# 4. `-v`가 하는 일: 복사가 아니라 창(窓)

`-v "$WS":/ws`가 하는 일은 **복사가 아니라 창(窓)을 내는 것**이다.

컨테이너 안에서

```bash
touch /ws/ros2/launch/foo.py
```

를 만들면, 호스트의

```text
~/local_ws/SeoulDynamics.SLAM/ros2/launch/foo.py
```

에 **그 순간** 생긴다. 반대도 마찬가지다.

```text
    호스트                        컨테이너
~/local_ws/.../ros2/  ◄────────►  /ws/ros2/
       │                              │
       └──────── 같은 inode ──────────┘
              복사본이 아니다
```

그래서 **VSCode로 호스트에서 편집하면서 컨테이너에서 빌드하는 것**이 가능하다.
파일을 옮기는 단계가 아예 없다.

---

# 5. `-w`는 시작 위치

```bash
-w /ws        # 컨테이너에 들어가면 여기서 시작
```

`-w`는 working directory다. 컨테이너에 들어가자마자 `cd`가 되어 있는 셈이다.

그래서 들어가자마자

```bash
./scripts/run.sh
```

가 바로 된다. `cd /ws`를 매번 칠 필요가 없다.

---

# 6. 확인해보는 방법

말로 이해하는 것보다 직접 확인하는 게 빠르다.

컨테이너에 들어가서:

```bash
ls /opt/ros/humble          # 이미지 안에만 있는 ROS
ls /ws                      # 호스트의 SeoulDynamics.SLAM
touch /ws/HELLO             # 만들고
```

호스트의 다른 터미널에서:

```bash
ls ~/local_ws/SeoulDynamics.SLAM/HELLO    # 바로 보임
```

반대 방향도 확인해보자. 호스트에서:

```bash
echo "from host" > ~/local_ws/SeoulDynamics.SLAM/HELLO2
```

컨테이너에서:

```bash
cat /ws/HELLO2              # from host
```

그리고 이건 안 보여야 정상이다.

```bash
ls /opt/ros/humble          # 호스트에서 실행하면 → No such file or directory
```

---

# 7. 왜 이 구분이 중요한가

이 구분을 놓치면 아래 착각을 하게 된다.

```text
착각 1
"컨테이너 안에서 apt install 했으니 다음에도 있겠지"
→ 없다. 이미지에 안 구워졌으므로 컨테이너를 지우면 사라진다.

착각 2
"이미지를 다시 빌드했으니 내 소스도 초기화되겠지"
→ 안 된다. /ws는 호스트 파일이라 이미지와 무관하다.

착각 3
"호스트에서 colcon build 한 install/을 컨테이너가 쓰면 되겠지"
→ 안 된다. 배포판도 라이브러리 ABI도 다르다.

착각 4
"컨테이너에서 만든 파일인데 왜 호스트에서 수정이 안 되지?"
→ 소유자가 root다. 파일은 보이지만 권한이 없다. (아래 “자주 겪는 문제 1”)
```

---

# 8. underlay와 overlay

ROS 2를 쓴다면 이 구분이 그대로 underlay / overlay와 겹친다.

```text
underlay  = /opt/ros/humble/
            이미지에 구워진 배포판. 안 바뀜.

overlay   = /ws/install/
            내가 colcon build로 만든 것. 매일 바뀜.
```

`source` 순서도 이 순서다.

```bash
source /opt/ros/humble/setup.bash    # underlay 먼저
source /ws/install/setup.bash        # overlay 나중
```

overlay를 나중에 source해야 같은 이름의 패키지가 있을 때 **내가 빌드한 것이 이긴다**.

---

# 9. `install/`이 `/ws` 아래인 이유

overlay는 매일 바뀌므로 이미지에 구울 수 없다.
그리고 **호스트에 남아야** 컨테이너를 껐다 켜도 다시 빌드하지 않아도 된다.

```text
/ws/install/  → 호스트에 남음 → 컨테이너 재시작해도 그대로
              → colcon build 10분을 매번 안 해도 됨
```

만약 `install/`이 컨테이너 안(예: `/root/install`)에 있었다면,
`docker run --rm`으로 컨테이너를 끌 때마다 빌드 결과가 통째로 날아간다.

---

# 10. `.gitignore`에 `install/`을 넣는 이유

`.gitignore`에 `install/`, `build/`, `log/`를 넣는 것도 같은 이유다.

```text
컨테이너가 만들지만 호스트에 남는 산출물
```

이기 때문이다. 호스트에 남으니 git이 보고,
하지만 빌드 산출물이라 커밋할 이유는 없다.

```gitignore
build/
install/
log/
```

---

# 11. 실제 파일 구성

정리하면 저장소는 보통 이런 모양이 된다.

```text
~/local_ws/SeoulDynamics.SLAM/
├── docker/
│   ├── Dockerfile        ← 이미지에 무엇을 구울지
│   ├── build.sh          ← 이미지 굽기 (1회, 10~20분)
│   └── run.sh            ← 컨테이너 열기 (매번, 1초)
├── ros2/                 ← 내 소스
├── slam/
├── config/
├── scripts/
├── build/                ← 빌드 중간물  (gitignore)
├── install/              ← overlay      (gitignore)
└── log/                  ←              (gitignore)
```

---

# 12. Dockerfile — 이미지에 무엇을 굽나

```dockerfile
FROM ros:humble-ros-base

# 이미지에 구워질 것들 — 호스트에는 안 생긴다
RUN apt-get update && apt-get install -y \
      ros-humble-velodyne-driver \
      ros-humble-realsense2-camera \
      ros-humble-tf2-ros \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Livox-SDK2 → /usr/local/lib
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/livox \
    && cmake -S /tmp/livox -B /tmp/livox/build \
    && cmake --build /tmp/livox/build --target install \
    && ldconfig \
    && rm -rf /tmp/livox

WORKDIR /ws
```

여기 적힌 것만 이미지에 들어간다.
**내 소스는 여기 없다.** 소스는 `COPY`하지 않고 `run.sh`에서 마운트한다.

---

# 13. `build.sh` — 한 번만, 10~20분

```bash
#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f docker/Dockerfile \
  -t sd-slam:humble \
  .
```

언제 다시 돌리나:

```text
Dockerfile을 고쳤을 때
새 apt 패키지가 필요할 때
base image를 바꿀 때
```

소스만 고쳤을 때는 **다시 돌릴 필요가 없다.**

---

# 14. `run.sh` — 컨테이너를 매번 새로 만들지 않는다

처음 쓰던 방식은 `docker run --rm`이었다. 나갈 때마다 컨테이너가 삭제되므로
안에서 `apt install`한 것이 전부 사라진다. `nano` 하나 쓰려고 매번 다시 설치하게 된다.

그래서 실전에서는 **없으면 만들고, 있으면 켜서 붙는** 방식을 쓴다.

```bash
#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
NAME=sd-slam
IMAGE=sd-slam:humble

# GUI(rviz2, rqt_graph)를 호스트 모니터에 띄우기 위한 권한
xhost +local: >/dev/null 2>&1 || true

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  # ① 컨테이너가 없다 → 생성. -e 옵션들은 "이 순간" 컨테이너에 박힌다
  docker run -dit \
    --name "$NAME" \
    --network host \
    --privileged \
    -v "$WS":/ws \
    -w /ws \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///ws/config/dds/cyclonedds-thor.xml \
    "$IMAGE" \
    bash
else
  # ② 이미 있다 → 꺼져 있으면 켜기만
  docker start "$NAME" >/dev/null
fi

# ③ 어느 경우든 마지막은 접속
docker exec -it "$NAME" bash
```

`-dit`의 `d`는 detached, 즉 백그라운드로 띄운다는 뜻이다.
컨테이너는 백그라운드에서 계속 살아 있고, 우리는 `docker exec`로 들어갔다 나왔다 한다.

```text
  처음 한 번          ./docker/build.sh     이미지 생성 (10~20분)
                     ./docker/run.sh       컨테이너 생성 + 접속

  그 다음부터         ./docker/run.sh       이미 있는 컨테이너에 접속 (1초)

  다른 터미널         docker exec -it sd-slam bash

  상태 보기           docker ps
  잠시 끄기           docker stop sd-slam
  아예 지우기         docker rm -f sd-slam   ← -e 값을 바꾸려면 이게 필요
```

이제 컨테이너 안에서 `apt install nano` 한 것이 다음에 들어가도 남아 있다.
다만 **계속 필요한 패키지는 여전히 Dockerfile에 넣는 것이 맞다.**
컨테이너에 직접 설치한 것은 `docker rm`을 하는 순간 사라지고, 팀원에게 공유되지도 않는다.

---

# 15. `run.sh`의 옵션 하나씩

```text
-dit             백그라운드로 띄우고 터미널을 붙일 수 있게 (detach+interactive+tty)
--name sd-slam   docker exec / stop / rm 에서 부를 이름
--network host   컨테이너가 호스트 네트워크를 그대로 쓴다. ROS 2 DDS discovery에 사실상 필수
--privileged     device 권한. 운영에서는 --device로 좁히는 게 낫다
-v "$WS":/ws     ★ 핵심. 호스트 작업공간을 /ws에 연결
-w /ws           들어가면 /ws에서 시작
-v /dev:/dev     LiDAR, 카메라 device node 접근

-v /tmp/.X11-unix:/tmp/.X11-unix   GUI 창을 호스트 화면에 그리기 위한 통로
-e DISPLAY                          어느 화면에 그릴지

-e ROS_DOMAIN_ID=42                 ROS 2 통신 영역 번호
-e RMW_IMPLEMENTATION=...           어떤 DDS 구현을 쓸지
-e CYCLONEDDS_URI=...               그 DDS가 어느 네트워크 인터페이스를 쓸지
```

---

# 16. `-e`는 이미지가 아니라 **컨테이너 생성 시** 박힌다

여기서 헷갈리기 쉽다. Dockerfile에 쓰는 것과 `-e`로 주는 것은 시점이 다르다.

```text
Dockerfile  ─ docker build ─→  이미지        ─ docker run ─→  컨테이너
   RUN apt-get install            (청사진)         -e ROS_DOMAIN_ID=42
   ENV ...                                          ↑
      ↑                                        여기서 박힌다
  여기서 박힌다
```

- **이미지에 박히는 것**: `RUN`으로 설치한 패키지, `ENV`로 넣은 기본값.
  `build.sh`를 다시 돌려야 바뀐다.
- **컨테이너에 박히는 것**: `docker run -e`로 준 환경변수.
  **`docker run`이 실행되는 그 순간 한 번** 결정된다.

컨테이너를 재사용하도록 바꿨으므로, `run.sh`의 `-e` 값을 고쳐도
이미 만들어진 컨테이너에는 반영되지 않는다. 값을 바꾸려면 컨테이너를 다시 만들어야 한다.

```bash
docker rm -f sd-slam     # 기존 컨테이너 삭제
./docker/run.sh          # 새 -e 값으로 다시 생성
```

이미지를 다시 빌드할 필요는 없다. 컨테이너만 다시 만들면 된다.

확인은 컨테이너 안에서:

```bash
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
env | grep -E 'ROS_|CYCLONE|DISPLAY'
```

---

# 17. `ROS_DOMAIN_ID` — ROS 2의 네트워크 영역 번호

ROS 2 노드는 같은 `ROS_DOMAIN_ID` 값을 가진 노드끼리만 서로를 발견하고 통신한다.
같은 랜에 있어도 번호가 다르면 서로 존재를 모른다.

```bash
# 터미널 1
export ROS_DOMAIN_ID=10
ros2 run my_pkg my_node

# 터미널 2
export ROS_DOMAIN_ID=10
ros2 topic list          # 노드가 보인다
```

번호가 다르면:

```text
터미널 1: ROS_DOMAIN_ID=10  →  노드 실행 중
터미널 2: ROS_DOMAIN_ID=20  →  ros2 topic list 가 비어 있음

  노드는 멀쩡히 돌고 있다. 서로 다른 ROS 네트워크에 있을 뿐이다.
```

토픽이 안 보일 때 가장 먼저 의심할 것이 이 값이다. 호스트에 고정하려면:

```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
source ~/.bashrc
echo $ROS_DOMAIN_ID
```

**단, 호스트의 환경변수는 컨테이너에 자동으로 전달되지 않는다.**
호스트 `~/.bashrc`에 넣어도 컨테이너 안에서는 비어 있다.
그래서 `run.sh`에 `-e ROS_DOMAIN_ID=42`가 필요하다.

두 대의 로봇 컴퓨터(예: Thor와 Xavier)가 ROS 2로 대화하려면 **양쪽 모두** 같은 값이어야 한다.

---

# 18. RMW와 DDS — `RMW_IMPLEMENTATION`은 무엇을 고르는 건가

ROS 2는 스스로 네트워크 통신을 구현하지 않는다. DDS라는 별도의 통신 미들웨어에 맡긴다.
그런데 DDS 제품이 여러 개라서, 그 사이에 얇은 어댑터 층을 하나 두었다. 그게 **RMW(ROS MiddleWare)**다.

```text
   ROS 2 노드 (rclcpp / rclpy)
        │
        ▼
   RMW — ROS가 정한 공통 인터페이스
        │
        ▼
   rmw_cyclonedds_cpp     ← 어댑터. 여기를 갈아끼운다
        │
        ▼
   CycloneDDS             ← 실제 통신을 하는 DDS 구현
        │
        ▼
   네트워크 → 상대 로봇
```

`RMW_IMPLEMENTATION`은 **어느 어댑터를 쓸지** 고르는 환경변수다.

```bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp      # Fast DDS  (ROS 2 기본값)
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp    # CycloneDDS
RMW_IMPLEMENTATION=rmw_connextdds        # RTI Connext DDS
```

`rmw_cyclonedds_cpp`로 설정한다는 것은 “ROS 2의 통신 계층으로 CycloneDDS를 써라”는 뜻이지,
그 자체가 상대 컴퓨터와 연결해주는 기능은 아니다. **양쪽이 같은 DDS 계열이어야 한다.**

설치되어 있는지 확인:

```bash
ros2 pkg prefix rmw_cyclonedds_cpp
```

Dockerfile에서 미리 깔아 둔다:

```dockerfile
RUN apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp
```

---

# 19. `CYCLONEDDS_URI` — 어느 랜 카드로 통신할지

로봇 컴퓨터에는 네트워크 인터페이스가 여러 개 붙어 있다.

```text
wlan0    Wi-Fi
eth0     LiDAR 전용 이더넷
eth1     상대 컴퓨터 연결용 이더넷
```

DDS가 알아서 고르게 두면 엉뚱한 인터페이스로 discovery를 시도해서
“핑은 되는데 토픽은 안 보이는” 상태가 된다. XML로 명시해준다.

```xml
<!-- config/dds/cyclonedds-thor.xml -->
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>eth1</NetworkInterfaceAddress>
    </General>
  </Domain>
</CycloneDDS>
```

그리고 그 파일을 가리킨다.

```bash
-e CYCLONEDDS_URI=file:///ws/config/dds/cyclonedds-thor.xml
```

경로가 `/ws/...`인 것에 주목하자. 이 XML은 **마운트된 호스트 파일**이므로
호스트에서 편집하고 컨테이너를 다시 만들기만 하면 된다. 이미지 재빌드가 필요 없다.

---

# 20. X11, `DISPLAY`, `xhost` — 컨테이너의 GUI를 호스트 모니터에 띄우기

`rviz2`나 `rqt_graph`는 창을 띄우는 프로그램이다. 그런데 컨테이너에는 모니터가 없다.
Linux에서 화면을 그리는 일은 X 서버가 하고, 컨테이너는 **호스트의 X 서버에 그려달라고 부탁**해야 한다.

```text
  컨테이너의 rviz2
        │  "이 창 좀 그려줘"
        ▼
  /tmp/.X11-unix   ← 소켓. -v 로 컨테이너 안에 연결해준다
        │
        ▼
  호스트의 X 서버 → 실제 모니터
```

필요한 것은 세 가지다.

```bash
-v /tmp/.X11-unix:/tmp/.X11-unix    # 통로를 연결
-e DISPLAY="$DISPLAY"               # 어느 화면인지 알려줌 (보통 :0)
xhost +local:                       # 호스트가 로컬 접속을 허용 (호스트에서 실행)
```

`xhost +local:`은 호스트 쪽에서 한 번 실행해야 하므로 `run.sh` 맨 위에 넣어 두었다.
빼먹으면 `Authorization required, but no authorization protocol specified` 같은 오류가 난다.

테스트:

```bash
# 컨테이너 안에서
apt-get update && apt-get install -y x11-apps
xeyes                    # 눈알 두 개가 호스트 화면에 뜨면 성공
```

`zenity`는 GUI 알림 창을 띄우는 작은 Linux 프로그램이다.
X11 설정이 되어 있으면 같이 동작하지만 필수는 아니다. **핵심은 `DISPLAY`와 X11 소켓 마운트 두 가지다.**

---

# 21. 서로 다른 ROS 2 배포판끼리 통신되는지 확인하기

Humble과 Galactic처럼 배포판이 달라도 ROS 2 통신 자체는 된다.
DDS는 배포판이 아니라 wire protocol을 맞추기 때문이다. 대신 아래가 맞아야 한다.

```text
① ROS_DOMAIN_ID       양쪽 같은 값
② DDS 구현            양쪽 같은 계열 (예: 둘 다 CycloneDDS)
③ 네트워크            서로 ping 되고, multicast가 막히지 않을 것
④ 메시지 패키지        같은 이름 + 같은 필드 정의
```

④가 조용히 실패하는 지점이다. 토픽 이름은 보이는데 `echo`가 안 되면 대개 메시지 정의 불일치다.

확인은 **아래에서 위로** 올라간다. 안 되는 지점에서 멈추면 그게 원인이다.

```bash
# 1) 네트워크가 되나
ping -c 3 192.168.1.20

# 2) 양쪽 ROS 환경이 같나  (양쪽 컴퓨터에서 각각)
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION

# 3) 상대 노드가 보이나
ros2 node list
ros2 topic list

# 4) 데이터가 실제로 오나
ros2 topic echo /mcu/state/d_imu --once
ros2 topic hz   /mcu/state/d_imu

# 5) 타입이 맞나  (여기서 깨지면 메시지 패키지 불일치)
ros2 topic type    /mcu/state/d_imu
ros2 interface show draper_msgs/msg/ImuBundle
```

4번까지 정상 출력되면 서로 다른 배포판 사이의 ROS 2 통신은 정상이라고 봐도 된다.

연결 관계를 그림으로 보고 싶으면:

```bash
apt-get update && apt-get install -y ros-humble-rqt-graph
rqt_graph
```

`ros2 topic list`는 목록만 보여주고 **누가 누구에게 보내는지는 알려주지 않는다.**
그 연결 방향을 보는 것이 `rqt_graph`다. (GUI이므로 §20의 X11 설정이 되어 있어야 한다.)

---

# 22. Docker에서 ROS 2가 안 보일 때

```text
증상                              먼저 볼 것
──────────────────────────────    ────────────────────────────────────
컨테이너 안에서 topic list 가     ROS_DOMAIN_ID 가 컨테이너에 전달됐나
비어 있음                         → env | grep ROS_DOMAIN_ID
                                  → run.sh 의 -e 확인

호스트에선 보이는데 컨테이너       --network host 를 안 썼다.
에선 안 보임                      기본 bridge 네트워크는 DDS multicast
                                  discovery 를 막는다

노드는 보이는데 echo 가 멈춤       메시지 타입 불일치, 또는 QoS 불일치
                                  → ros2 topic type / interface show

한쪽에서만 보임                    DDS 구현이 서로 다름
                                  → 양쪽 echo $RMW_IMPLEMENTATION

인터페이스가 여러 개인데 불안정     CYCLONEDDS_URI 로 인터페이스 고정
```

`--network host`는 편의 옵션이 아니라 **ROS 2에서는 사실상 기본 설정**으로 생각하는 편이 낫다.

---

# 23. 하루 작업 흐름

```bash
# 1. 컨테이너 열기 (호스트 터미널)
cd ~/local_ws/SeoulDynamics.SLAM
./docker/run.sh

# ── 여기부터 컨테이너 안 ──

# 2. underlay는 base image에서 이미 잡혀 있는 경우가 많다. 확인:
echo $ROS_DISTRO          # humble

# 3. 빌드 (overlay 생성)
colcon build --symlink-install

# 4. overlay source
source install/setup.bash

# 5. 실행
ros2 launch my_slam bringup.launch.py
```

편집은 호스트 VSCode에서 하면 된다.
저장하는 순간 컨테이너의 `/ws`에도 반영되므로,
컨테이너 터미널에서 `colcon build`만 다시 돌리면 된다.

---

# 24. 터미널을 여러 개 열고 싶을 때

`run.sh`를 두 번 돌리면 **컨테이너가 두 개** 뜬다. 그건 보통 원하는 게 아니다.
이미 떠 있는 컨테이너에 붙으려면 `docker exec`를 쓴다.

```bash
docker exec -it sd-slam bash
```

들어가서 overlay를 다시 source해야 한다.

```bash
cd /ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic list
```

`docker exec`로 들어간 shell은 `run.sh`의 shell과 별개 프로세스라
환경변수를 물려받지 않기 때문이다.

---

# 25. 컨테이너 정리

```bash
docker ps                       # 실행 중
docker ps -a                    # 종료된 것 포함
docker stop sd-slam             # 정지
docker rm sd-slam               # 삭제 (--rm 안 썼을 때)
docker images                   # 이미지 목록
docker rmi sd-slam:humble       # 이미지 삭제 → build.sh 다시 필요
```

`docker rmi`를 해도 `/ws`는 안 지워진다. 다시 강조할 만한 지점이다.

---

# 26. 자주 겪는 문제 1 — 컨테이너가 만든 파일이 root 소유

컨테이너 안에서 기본 사용자는 root다.
그래서 `colcon build`가 만든 `install/`이 호스트에서 root 소유로 보인다.

```bash
ls -l ~/local_ws/SeoulDynamics.SLAM
# drwxr-xr-x root root install
```

호스트 VSCode에서 지우거나 수정하려 하면 권한 오류가 난다.

해결 1 — 컨테이너를 내 UID로 실행:

```bash
docker run -it --rm \
  --user "$(id -u):$(id -g)" \
  -v "$WS":/ws -w /ws \
  sd-slam:humble bash
```

해결 2 — 이미 생긴 것을 되돌리기:

```bash
sudo chown -R "$(id -u):$(id -g)" ~/local_ws/SeoulDynamics.SLAM
```

`--user`를 쓰면 `/opt/ros` 같은 이미지 내부 경로에 쓰기가 막힐 수 있으므로,
개발 컨테이너에서는 Dockerfile에 같은 UID의 non-root 사용자를 만들어 두는 방식도 많이 쓴다.

---

# 27. 자주 겪는 문제 2 — 호스트에서 빌드한 `install/`을 컨테이너가 못 씀

호스트(noble)에서 `colcon build`를 한 적이 있으면
그 `install/`이 `/ws/install/`로 그대로 보인다. 그런데 실행하면 깨진다.

```text
호스트 noble에서 링크된 libstdc++ / libssl
        ≠
컨테이너 jammy의 것
```

같은 파일이 보인다고 같은 환경에서 만들어진 게 아니다.

```bash
# 컨테이너 안에서
rm -rf build install log
colcon build --symlink-install
```

**빌드는 항상 한쪽에서만** 하는 것이 규칙이다.

---

# 28. 자주 겪는 문제 3 — 이미지를 다시 빌드했는데 `/ws`가 그대로

정상이다. 의도한 동작이다.

```text
build.sh  → 이미지만 바꾼다
/ws       → 호스트 파일. 이미지와 무관
```

새 이미지의 라이브러리에 맞춰 다시 빌드하고 싶으면 `/ws` 쪽을 직접 지워야 한다.

```bash
rm -rf build install log && colcon build --symlink-install
```

---

# 29. 자주 겪는 문제 4 — `apt install`한 게 사라짐

컨테이너 안에서

```bash
apt-get install -y ros-humble-something
```

를 했는데 다음에 들어가 보니 없다. 어디에 설치됐는지가 문제다.

```text
설치한 곳          컨테이너 레이어 (이미지가 아니다)

docker stop/start  → 남아 있다  ✓
docker rm          → 사라진다   ✗
build.sh 재실행    → 새 이미지에는 애초에 없다  ✗
다른 사람 PC       → 당연히 없다  ✗
```

컨테이너를 재사용하도록 바꿨으므로 `stop` / `start`로는 살아남는다.
그래도 `docker rm`을 하는 순간, 그리고 팀원에게 넘기는 순간 전부 사라진다.

```text
한 번 쓰고 말 것 (디버깅용 tcpdump 등)  → 컨테이너에서 그냥 설치
계속 필요한 것 (nano, rqt_graph, ...)   → Dockerfile에 추가 → build.sh
매일 바뀌는 내 코드                      → /ws (마운트)
```

“매번 다시 깔기 귀찮다”의 진짜 해법은 컨테이너를 살려두는 것이 아니라
**Dockerfile에 한 줄 추가하고 이미지를 다시 굽는 것**이다.

```dockerfile
RUN apt-get update && apt-get install -y \
      nano iproute2 iputils-ping tcpdump \
      ros-humble-rqt-graph \
      ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
```

---

# 30. 무엇을 이미지에 굽고 무엇을 마운트할까

```text
이미지에 굽는다 (Dockerfile)
├── OS, 배포판
├── ROS 2 배포판, 드라이버 패키지
├── CUDA / TensorRT / Rust toolchain
├── Livox-SDK2 같은 외부 SDK
└── → 바뀌는 주기: 몇 주 ~ 몇 달

마운트한다 (-v)
├── 내 소스 (ros2/, slam/, config/)
├── 빌드 산출물 (build/, install/, log/)
├── 로그, bag 파일, 맵
└── → 바뀌는 주기: 몇 분
```

기준은 단순하다. **바뀌는 주기가 짧으면 마운트, 길면 이미지.**

---

# 31. Mini Practice 1

컨테이너 안에서 호스트와 배포판이 다른지 확인한다.

```bash
# 호스트
cat /etc/os-release | head -2

# 컨테이너
./docker/run.sh
cat /etc/os-release | head -2
uname -r                    # 커널은 호스트 것과 같다
```

커널 버전만 같고 배포판은 다르다는 것을 확인한다.

---

# 32. Mini Practice 2

양방향 반영을 확인한다.

```bash
# 컨테이너에서
touch /ws/HELLO

# 호스트 다른 터미널에서
ls ~/local_ws/SeoulDynamics.SLAM/HELLO
echo edited > ~/local_ws/SeoulDynamics.SLAM/HELLO

# 다시 컨테이너에서
cat /ws/HELLO               # edited
rm /ws/HELLO
```

---

# 33. Mini Practice 3

이미지에만 있는 것과 마운트된 것의 수명을 비교한다.

```bash
# 컨테이너에서
touch /opt/MARKER_IMAGE     # 컨테이너 레이어에 씀
touch /ws/MARKER_MOUNT      # 호스트에 씀
exit

# ① stop/start 로는 둘 다 살아남는다
docker stop sd-slam && ./docker/run.sh
ls /opt/MARKER_IMAGE        # ✓ 있다
ls /ws/MARKER_MOUNT         # ✓ 있다
exit

# ② 컨테이너를 지우면 갈린다
docker rm -f sd-slam
./docker/run.sh
ls /opt/MARKER_IMAGE        # ✗ 없다 — 컨테이너와 함께 사라짐
ls /ws/MARKER_MOUNT         # ✓ 있다 — 호스트 파일이므로
```

②가 §3의 표를 그대로 재현한다.

---

# 34. Mini Practice 4

컨테이너에 ROS 2 환경변수가 실제로 전달됐는지 확인한다.

```bash
# 컨테이너 안에서
env | grep -E 'ROS_|CYCLONE|DISPLAY'
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
ros2 pkg prefix rmw_cyclonedds_cpp

# 값을 바꿔보고 싶다면 — 컨테이너를 다시 만들어야 한다
exit
docker rm -f sd-slam
# run.sh 의 -e ROS_DOMAIN_ID 를 고친 뒤
./docker/run.sh
echo $ROS_DOMAIN_ID          # 바뀐 값
```

이미지는 다시 빌드하지 않았는데도 값이 바뀐다는 점을 확인한다.

---

# 35. 오늘의 핵심

```text
          컨테이너에서 보이는 경로

    ┌───────────────────────────────┐
    │  /opt/ros/humble              │
    │  /usr/local/lib               │  ← 이미지 레이어
    │  /usr/local/cargo             │     build.sh가 만듦
    │                               │     컨테이너 지우면 사라짐
    ├───────────────────────────────┤
    │  /ws                          │  ← bind mount
    │    ros2/ slam/ config/        │     호스트 파일 그 자체
    │    build/ install/ log/       │     컨테이너와 무관하게 남음
    └───────────────────────────────┘
              │
              ▼
        -v "$WS":/ws
      복사가 아니라 창(窓)
```

---

# 36. 반드시 구분할 것

```text
이미지 레이어 ≠ Bind Mount

/opt/ros/humble (underlay)
≠
/ws/install (overlay)

build.sh (1회, 10~20분)
≠
run.sh (매번, 1초)

컨테이너 삭제
≠
/ws 삭제

호스트에서 빌드한 install/
≠
컨테이너에서 쓸 수 있는 install/

-v (창을 냄)
≠
COPY (이미지에 복사)

docker run (새 컨테이너)
≠
docker exec (기존 컨테이너)

docker stop  (껐다 켤 수 있음)
≠
docker rm    (설치한 것 다 날아감)

Dockerfile ENV (이미지에 박힘)
≠
docker run -e (컨테이너 생성 시 박힘)

RMW (ROS의 추상화 계층)
≠
DDS (실제 통신 구현)

RMW_IMPLEMENTATION (어느 DDS를 쓸지)
≠
ROS_DOMAIN_ID (어느 영역에서 통신할지)
≠
CYCLONEDDS_URI (어느 랜 카드를 쓸지)

ros2 topic list (목록만)
≠
rqt_graph (누가 누구에게 보내는지)

호스트의 환경변수
≠
컨테이너의 환경변수  (자동으로 안 넘어간다)
```

---

# 37. Chapter 연결

```text
Chapter 6
ROS 2 — workspace, colcon, underlay/overlay

Chapter 6.5
micro-ROS — Agent를 컨테이너에서 돌릴 때 --device 가 필요한 이유

Chapter 8
Robot Networking — DDS, --network host가 필요한 이유

Chapter 9
Docker on Jetson — image, container, bind mount 개념

Chapter 9.5  ← 여기
컨테이너 파일시스템과 개발 워크플로우

Chapter 10
Debugging & Deployment — 이 구조에서 문제가 났을 때 어디부터 보나
```

Chapter 9가 "Docker란 무엇인가"라면
이 chapter는 "그래서 내 작업공간은 어디에 있는가"에 해당한다.

Chapter 10에서는 여기서 만든 컨테이너가 실제로 안 돌 때
Hardware → Linux → Network → Docker → ROS 2 → CUDA → Application 순서로 좁혀 들어간다.
