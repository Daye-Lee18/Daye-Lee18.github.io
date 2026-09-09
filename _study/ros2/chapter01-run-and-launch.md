---
title: "Chapter 1. ros2 run과 ros2 launch"
importance: 2
---

> **Goal:** `ros2 run <package> <executable>`의 두 인자가 각각 어디서 온 이름인지 알고,
> 설정을 `run`에 직접 넘기는 방법과 `launch` 파일에 적는 방법을 구분해서 쓸 수 있다.

`ros2 run fast_lio fastlio_mapping` 같은 명령이 헷갈리는 이유는
**두 이름이 서로 다른 곳에서 오기 때문**이다. 그걸 알면 헷갈릴 일이 없어진다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `ros2 pkg executables` · `ros2 run` · `ros2 launch`

| 명령어                                                 | 하는 일                             |
| :----------------------------------------------------- | :---------------------------------- |
| `ros2 pkg list`                                        | 지금 환경에서 보이는 패키지 전부    |
| `ros2 pkg executables`                                 | 전체 패키지의 실행 파일             |
| `ros2 pkg executables <pkg>`                           | **그 패키지가 뭘 실행할 수 있는지** |
| `ros2 pkg prefix <pkg>`                                | 그 패키지가 어디서 왔는지           |
| `ros2 run <pkg> <exe>`                                 | 실행 파일 하나 실행                 |
| `ros2 run <pkg> <exe> --ros-args -p k:=v`              | 파라미터 하나 넘기기                |
| `ros2 run <pkg> <exe> --ros-args --params-file f.yaml` | 파라미터 파일                       |
| `ros2 run <pkg> <exe> --ros-args -r a:=b`              | 토픽 이름 바꾸기                    |
| `ros2 launch <pkg> <file>.launch.py`                   | launch 파일 실행                    |
| `ros2 launch <pkg> <file>.launch.py arg:=v`            | launch 인자                         |
| `ros2 launch -s <pkg> <file>.launch.py`                | **그 launch가 받는 인자 목록**      |
| `ros2 launch -p <pkg> <file>.launch.py`                | 실행 안 하고 내용만 출력            |
| `ros2 node list` / `ros2 topic list`                   | 실행 결과 확인                      |

---

# 1. `ros2 run`의 두 인자

```bash
ros2 run fast_lio fastlio_mapping
        └───┬───┘ └──────┬──────┘
         패키지 이름    실행 파일 이름
```

두 이름의 출처가 다르다.

```text
패키지 이름     package.xml 의 <name>
실행 파일 이름   C++  : CMakeLists.txt 의 add_executable() + install()
               Python: setup.py 의 entry_points
```

**폴더 이름이 아니다.** 폴더가 `src/slam/`이어도 `package.xml`에 `slam_ros2`라고 적혀 있으면
`ros2 run slam_ros2 ...`다. (Edge Computing Chapter 6.2 참고)

외울 필요 없이 물어보면 된다.

```bash
$ ros2 pkg executables fast_lio
fast_lio fastlio_mapping
└──┬───┘ └──────┬──────┘
  1번 인자     2번 인자
```

**출력이 그대로 `ros2 run` 뒤에 들어갈 형태**다. 이게 가장 빠른 확인법이다.

아무것도 안 나오면 셋 중 하나다.

```text
· source 를 안 했다
· 빌드가 안 됐다
· CMakeLists 에 install(TARGETS ...) 이 없다   ← 빌드는 성공하는 함정
```

---

# 2. `ros2 run`은 설정 파일을 스스로 찾지 않는다

가장 흔한 오해다. **`ros2 run`은 config 파일을 자동으로 읽지 않는다.**
아무것도 안 주면 코드에 박힌 기본값으로 돈다.

```text
ros2 run <pkg> <exe>
     │
     └── 실행 파일 하나를 그냥 실행한다. 그게 전부다
         설정을 원하면 명령줄로 직접 준다
```

설정은 `--ros-args` 뒤에 붙인다.

```bash
# 파라미터 파일
ros2 run slam_ros2 estimator --ros-args --params-file config/slam.yaml

# 파라미터 하나만
ros2 run slam_ros2 estimator --ros-args -p max_range:=100.0

# 토픽 이름 바꾸기 (remapping)
ros2 run slam_ros2 estimator --ros-args -r input:=/sensors/imu/data

# 여러 개 (--ros-args 는 한 번만)
ros2 run slam_ros2 estimator --ros-args \
  --params-file config/slam.yaml \
  -p use_sim_time:=true \
  -r /odom:=/mcu/command/odom
```

`--ros-args`가 경계선이다. **그 앞은 실행할 것, 뒤는 ROS 설정.**

```text
ros2 run  pkg  exe   --ros-args   -p ...  -r ...
          └── 무엇을 ──┘   ↑      └── 어떻게 ──┘
                       경계선
```

---

# 3. `ros2 launch`는 파일에 적어 둔다

노드가 하나면 §2로 충분하다. 그런데 실제 시스템은 이렇다.

```text
LiDAR 드라이버 + IMU 드라이버 + SLAM + TF + rviz
각각 파라미터 파일과 remapping 이 다르고
띄우는 순서도 있다
```

이걸 매번 손으로 치는 대신 파일에 적는 것이 launch다.

```bash
ros2 launch slam_ros2 bringup.launch.py
```

```python
# launch/bringup.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('slam_ros2'),   # ← install/share/slam_ros2
        'config', 'slam.yaml')

    return LaunchDescription([
        Node(
            package='slam_ros2',
            executable='estimator',       # ← ros2 run 의 2번 인자와 같은 이름
            name='slam_estimator',        # ← 노드 이름을 여기서 정한다
            parameters=[cfg],
            remappings=[('input', '/sensors/imu/data')],
            output='screen',
        ),
        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            parameters=[{'device_ip': '192.168.1.201'}],
        ),
    ])
```

`ros2 run`에서 명령줄로 주던 것이 그대로 필드가 된다.

```text
ros2 run 의 것                launch 의 것
──────────────────────       ──────────────────
<package>                    package=
<executable>                 executable=
--ros-args --params-file     parameters=[...]
--ros-args -r a:=b           remappings=[...]
(노드 이름은 코드에서)          name=
```

`get_package_share_directory()`가 중요하다.
**`install/<pkg>/share/<pkg>/`를 가리킨다.** 상대 경로를 쓰면 어디서 실행하느냐에 따라 깨진다.

launch가 받는 인자를 모를 때는 물어보면 된다.

```bash
ros2 launch -s slam_ros2 bringup.launch.py    # 받을 수 있는 인자 목록
ros2 launch slam_ros2 bringup.launch.py use_rviz:=true
```

---

# 4. 파일이 실제로 있는 곳 — `src/`가 아니라 `install/`

여기서 한 번 크게 걸린다.

```text
내가 편집하는 곳                     ROS 가 읽는 곳
src/slam_ros2/launch/bringup.py     install/slam_ros2/share/slam_ros2/launch/bringup.py
src/slam_ros2/config/slam.yaml      install/slam_ros2/share/slam_ros2/config/slam.yaml
```

`ros2 launch`는 **`install/` 쪽을 읽는다.** `src/`를 고쳐도 `install/`이 안 바뀌면 반영되지 않는다.

그래서 `colcon build`에 `install()` 규칙이 있어야 한다. (Chapter 6.2 §8)

```cmake
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})
```

이게 없으면 `ros2 launch`가 파일을 못 찾는다. **빌드는 성공한다.**

---

# 5. 그러면 launch 파일을 고칠 때마다 다시 빌드해야 하나

`--symlink-install`을 썼는지에 따라 다르다. colcon 문서의 설명은 이렇다.

```text
--symlink-install
    Use symlinks instead of copying files from the source
    and build directories where possible.
```

```text
colcon build                     install/ 에 복사본이 생긴다
                                 → launch/config 를 고치면 재빌드 필요

colcon build --symlink-install   install/ 이 src/ 를 가리킨다 (심볼릭 링크)
                                 → 저장하는 순간 반영. 재빌드 불필요
```

확인은 눈으로 된다.

```bash
$ ls -l install/slam_ros2/share/slam_ros2/launch/
lrwxr-xr-x  bringup.launch.py -> /home/me/ws/src/slam_ros2/launch/bringup.launch.py
                              ↑ 화살표가 있으면 심볼릭 링크. 재빌드 불필요
```

**그래서 `--symlink-install`을 거의 항상 쓴다.**
launch·config·Python 노드를 고치면서 매번 빌드를 기다릴 이유가 없다.

C++ 소스는 예외다. 컴파일이 필요하므로 무조건 다시 빌드해야 한다.

```text
고친 것              --symlink-install 이면
─────────────       ──────────────────────
launch/*.py          재빌드 불필요
config/*.yaml        재빌드 불필요
Python 노드          재빌드 불필요
C++ 소스             재빌드 필요 (컴파일)
CMakeLists/setup.py  재빌드 필요
package.xml          재빌드 필요
```

---

# 6. `source` 두 줄의 정확한 의미

```bash
source /opt/ros/humble/setup.bash   # ROS 2 배포판 (underlay)
source ~/ros2_ws/install/setup.bash # 이 workspace 에서 빌드한 패키지 (overlay)
```

두 번째 줄을 "install 패키지 환경"이라고 하면 조금 어긋난다.
정확히는 **"이 workspace에서 빌드해서 설치한 패키지"**다. `install/`이라는 폴더 이름 때문에
apt로 설치한 것처럼 들리는데 그게 아니다.

그리고 실은 **두 번째 줄만 해도 되는 경우가 많다.**

```text
install/setup.bash        빌드 시점의 underlay 까지 같이 source 한다 (chaining)
install/local_setup.bash  이 workspace 것만
```

```bash
source ~/ros2_ws/install/setup.bash   # /opt/ros/humble 도 같이 딸려온다
```

`.bashrc`에 넣어두면 매번 안 쳐도 된다. 다만 workspace를 여러 개 오가면 오히려 헷갈리므로,
프로젝트별 스크립트를 두는 편이 낫다.

```bash
# scripts/env.sh
source /opt/ros/humble/setup.bash
source "$(dirname "$0")/../install/setup.bash"
export ROS_DOMAIN_ID=42
```

**source를 안 하면 증상이 이렇게 나온다.**

```text
$ ros2 pkg executables slam_ros2
Package not found

$ ros2 run slam_ros2 estimator
Package 'slam_ros2' not found
```

"방금 빌드했는데 없다고 나온다"의 대부분이 이것이다.
새 터미널을 열면 환경이 초기화되므로 다시 source해야 한다.

---

# 7. 실행 후 확인

```bash
ros2 node list                    # 노드가 떴나
ros2 node info /slam_estimator    # 그 노드의 토픽·서비스·파라미터
ros2 topic list                   # 토픽이 생겼나
ros2 topic hz /odom               # 실제로 데이터가 흐르나
ros2 topic info /odom --verbose   # 누가 publish/subscribe 하나, QoS 는
```

`ros2 node info`가 특히 유용하다. **remapping이 의도대로 됐는지**가 여기서 보인다.

```text
$ ros2 node info /slam_estimator
  Subscribers:
    /sensors/imu/data: sensor_msgs/msg/Imu     ← remapping 이 먹었다
                                                 (원래 이름은 input 이었다)
```

---

# 8. 언제 무엇을 쓰나

```text
ros2 run       노드 하나를 빠르게 띄워볼 때
               파라미터 하나만 바꿔서 테스트할 때
               "이 실행 파일이 뜨긴 하나" 확인할 때

ros2 launch    시스템을 실제로 돌릴 때
               노드가 둘 이상일 때
               설정을 파일로 남겨야 할 때 (재현성)
```

**개발 중에는 `run`, 운영에서는 `launch`**가 대략의 기준이다.
`launch`로 정리해 두면 그 파일 자체가 "이 시스템은 이렇게 뜬다"는 문서가 된다.

---

# 9. 안 될 때

```text
증상                                  먼저 볼 것
──────────────────────────────        ────────────────────────────────
Package 'x' not found                 source 했나 (§6)
                                      ros2 pkg list | grep x

No executable found                   ros2 pkg executables x
                                      install(TARGETS ...) 이 있나 (§4)
                                      Python 이면 entry_points

launch 파일을 못 찾음                   install(DIRECTORY launch ...) 이 있나
                                      ls install/<pkg>/share/<pkg>/launch/

launch 를 고쳤는데 반영 안 됨            --symlink-install 로 빌드했나 (§5)
                                      ls -l 로 화살표 확인

파라미터가 안 먹음                      노드가 declare_parameter 했나
                                      YAML 의 노드 이름이 실제와 같나 (Ch 2)

노드는 떴는데 토픽이 안 보임             remapping 확인: ros2 node info
                                      ROS_DOMAIN_ID (Edge Computing Ch 9.5)
```

---

# 10. Mini Practice

```bash
# 1) 무엇을 실행할 수 있는지 물어본다
ros2 pkg list | head
ros2 pkg executables demo_nodes_cpp

# 2) 그대로 실행
ros2 run demo_nodes_cpp talker
#    다른 터미널에서
ros2 topic list
ros2 topic echo /chatter --once

# 3) remapping 을 걸어본다
ros2 run demo_nodes_cpp talker --ros-args -r /chatter:=/my_chatter
ros2 topic list          # /my_chatter 로 바뀌었나

# 4) node info 로 확인
ros2 node info /talker

# 5) source 안 한 새 터미널에서
ros2 pkg executables demo_nodes_cpp    # 어떻게 되나?

# 6) launch 파일 위치 확인
ros2 pkg prefix demo_nodes_cpp
ls $(ros2 pkg prefix demo_nodes_cpp)/share/demo_nodes_cpp/launch/ 2>/dev/null
```

3번과 5번이 핵심이다.
**remapping은 코드를 안 고치고 연결을 바꾸는 방법**이고,
**source는 매 터미널마다 필요하다.**

---

# 11. 오늘의 핵심

```text
   package.xml <name>          ─────┐
   CMakeLists install(TARGETS)  ────┤
   setup.py entry_points        ────┤
                                    ▼
                        ros2 run <package> <executable>
                                    │
                                    │  --ros-args
                                    │     --params-file  파라미터
                                    │     -p k:=v        파라미터 하나
                                    │     -r a:=b        토픽 이름
                                    ▼
                              노드 하나 실행


   launch/*.launch.py  ──▶  ros2 launch <package> <file>
        │                        여러 노드 + 파라미터 + remapping
        │                        install/<pkg>/share/<pkg>/launch/ 를 읽는다
        └── --symlink-install 이면 src/ 를 가리키므로 재빌드 불필요
```

---

# 12. 반드시 구분할 것

```text
패키지 이름  ≠  폴더 이름
   package.xml 의 <name> 이 진짜

ros2 run 은 config 를 자동으로 안 읽는다
   --ros-args 로 직접 준다

src/  ≠  install/
   ROS 가 읽는 것은 install/<pkg>/share/<pkg>/

--symlink-install 있음  ≠  없음
   launch/config 수정에 재빌드가 필요한지가 갈린다

install/  ≠  apt 로 설치한 것
   "이 workspace 에서 빌드해 설치한 것"

setup.bash  ≠  local_setup.bash
   전자는 underlay 까지 딸려온다

빌드 성공  ≠  ros2 run 가능
   install(TARGETS ...) 과 source 가 사이에 있다

ros2 run  vs  ros2 launch
   하나 vs 여러 개 + 설정을 파일로 남기기
```

---

# 13. Chapter 연결

```text
Chapter 1  ← 여기
run 과 launch

Chapter 2
파라미터 — YAML 을 어디에 적고 누가 읽나

Chapter 3
remapping 과 namespace — 토픽 이름이 정해지는 규칙

Edge Computing Chapter 6
node, topic, DDS — 무엇이 도는가

Edge Computing Chapter 6.2
package.xml / CMakeLists / setup.py / colcon — 실행 파일 이름이 정해지는 곳
```
