---
title: "Chapter 6.2. ROS 2 파일 시스템과 빌드 시스템"
importance: 7.2
---

> **Goal:** `/opt/ros`와 내 workspace가 각각 무엇을 담고 있는지 구분하고,
> `package.xml` / `CMakeLists.txt` / `setup.py`가 각자 무슨 일을 하는지 알고,
> `colcon build`가 `src/`를 어떻게 `build/` · `install/`로 바꾸는지 설명할 수 있다.

Chapter 6은 ROS 2가 **무엇인지**를 다뤘다 — node, topic, DDS, QoS.
이 chapter는 그것들이 **어디에 놓이고 어떻게 만들어지는지**를 다룬다.

`colcon build` 한 줄만 외워서 쓰다가, 빌드가 깨지거나 패키지를 새로 만들어야 할 때
무엇을 봐야 할지 몰라 막히는 지점이 여기다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `colcon build --symlink-install` · `source install/setup.bash` · `rosdep install`

| 명령어                                                                  | 하는 일                                                       |
| :---------------------------------------------------------------------- | :------------------------------------------------------------ |
| `ros2 pkg create <name> --build-type ament_cmake --dependencies rclcpp` | C++ 패키지 생성                                               |
| `ros2 pkg create <name> --build-type ament_python --dependencies rclpy` | Python 패키지 생성                                            |
| `ros2 pkg list`                                                         | 지금 환경에서 보이는 패키지 전부                              |
| `ros2 pkg prefix <pkg>`                                                 | **그 패키지가 어디서 왔는지** (/opt/ros 인지 내 install 인지) |
| `ros2 pkg xml <pkg>`                                                    | 그 패키지의 package.xml                                       |
| `colcon build --symlink-install`                                        | 전체 빌드 (거의 항상 이 옵션)                                 |
| `colcon build --packages-select <pkg>`                                  | 그 패키지만                                                   |
| `colcon build --packages-up-to <pkg>`                                   | 그 패키지 + 의존하는 것들까지                                 |
| `colcon build --packages-above <pkg>`                                   | 그 패키지 + **그것에 의존하는** 것들                          |
| `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release`                  | 최적화 빌드                                                   |
| `colcon list`                                                           | workspace 안의 패키지 목록                                    |
| `colcon graph`                                                          | 의존 관계 그래프                                              |
| `colcon test` / `colcon test-result --verbose`                          | 테스트                                                        |
| `rosdep install --from-paths src --ignore-src -r -y`                    | **의존 패키지 한 번에 설치**                                  |
| `vcs import src < repos.yaml`                                           | 여러 저장소를 한 번에 clone                                   |
| `source install/setup.bash`                                             | 빌드 결과를 현재 shell에 적용                                 |

---

# 1. 두 개의 파일 시스템

Chapter 9.5에서 본 `/opt/ros`와 `/ws`의 구분이 여기서도 그대로다.
Docker를 안 쓰더라도 이 둘은 처음부터 별개다.

```text
배포판이 설치한 것                      내가 만든 것
/opt/ros/humble/                       ~/ros2_ws/
  apt 로 설치됨                          내가 소스에서 빌드
  건드리지 않는다                         매일 바뀐다
  underlay                              overlay
```

**패키지 하나가 어느 쪽에서 왔는지 헷갈릴 때** 이 명령이 답을 준다.

```bash
$ ros2 pkg prefix rclcpp
/opt/ros/humble

$ ros2 pkg prefix my_slam
/home/user/ros2_ws/install/my_slam
```

같은 이름의 패키지가 양쪽에 있으면 **나중에 source한 쪽이 이긴다.** (§14)

---

# 2. `/opt/ros/humble` 안에는 무엇이 있나

```text
/opt/ros/humble/
├── bin/              실행 파일 (ros2, rviz2, …)
├── lib/              라이브러리 (.so) 와 패키지별 실행 파일
├── include/          C++ 헤더
├── share/            패키지별 리소스
│   └── <pkg>/
│       ├── package.xml
│       ├── cmake/            다른 패키지가 find_package 할 때 쓴다
│       ├── msg/ srv/ action/ 인터페이스 정의
│       └── launch/ config/
├── setup.bash        환경 설정 (이걸 source 한다)
└── local_setup.bash  이 공간만 설정 (§14)
```

`share/`가 중요하다. **다른 패키지가 내 패키지를 찾을 수 있게 해주는 정보**가 여기 들어간다.
빌드했는데 `find_package()`가 실패한다면 대개 `share/<pkg>/cmake/`가 안 만들어진 것이다.

---

# 3. Package 하나의 구조

패키지는 ROS 2의 **배포·빌드 기본 단위**다.

```text
my_slam/
├── package.xml            ← 무엇에 의존하는지, 어떻게 빌드하는지 (필수)
├── CMakeLists.txt         ← C++ 패키지면 (ament_cmake)
├── setup.py / setup.cfg   ← Python 패키지면 (ament_python)
├── src/                   소스
├── include/my_slam/       공개 헤더
├── launch/                launch 파일
├── config/                YAML 파라미터
├── msg/  srv/  action/    인터페이스 정의
└── test/
```

**`package.xml`은 어느 쪽이든 반드시 있다.** 그게 "이건 ROS 2 패키지다"의 표시다.

관련된 패키지를 묶어서 한 번에 설치하게 만든 것이 **meta-package**다.
소스가 없고 `package.xml`에 의존성 목록만 들어 있다.
`ros-humble-desktop` 같은 것이 그런 식이다.

---

# 4. `package.xml` — 무엇을 선언하나

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_slam</name>
  <version>0.1.0</version>
  <description>SLAM node for Vision60</description>
  <maintainer email="me@example.com">me</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>   <!-- 빌드 도구 -->

  <depend>rclcpp</depend>                            <!-- 빌드 + 실행 둘 다 -->
  <depend>sensor_msgs</depend>
  <build_depend>eigen</build_depend>                 <!-- 빌드할 때만 -->
  <exec_depend>rviz2</exec_depend>                   <!-- 실행할 때만 -->
  <test_depend>ament_lint_auto</test_depend>         <!-- 테스트할 때만 -->

  <export>
    <build_type>ament_cmake</build_type>             <!-- ★ 빌드 방식 -->
  </export>
</package>
```

의존성 종류를 나눠 두는 이유가 있다.

```text
build_depend   컴파일에만 필요. 배포 이미지에는 없어도 된다
exec_depend    실행에만 필요. 헤더는 필요 없다
depend         둘 다
test_depend    테스트에만
```

**`colcon`은 이 선언을 읽어서 빌드 순서를 정하고, `rosdep`은 이걸 읽어서 설치한다.** (§10)
여기에 안 적으면 내 컴퓨터에서는 되는데 CI나 다른 사람 컴퓨터에서 깨진다.

---

# 5. Build System과 Build Tool은 다르다

용어가 헷갈리는데, 담당 범위가 다르다.

```text
Build System (빌드 시스템)         패키지 "하나"를 어떻게 빌드할지
   ament_cmake / ament_python      컴파일, 링크, 설치 위치

Build Tool (빌드 도구)             패키지 "여러 개"를 어떤 순서로 빌드할지
   colcon                          의존성 그래프 → 위상 정렬 → 병렬 실행
```

```text
        colcon                     ← build tool. 순서를 정한다
          │
   ┌──────┼──────┬──────────┐
   ▼      ▼      ▼          ▼
 pkg A  pkg B  pkg C  …           ← 각각을 build system 이 빌드한다
 ament_ ament_ ament_
 cmake  python cmake
```

역사적으로 ROS 1은 `catkin`(빌드 시스템)과 `catkin_make`/`catkin_tools`(도구)를 썼고,
ROS 2는 `ament`(빌드 시스템)와 `colcon`(도구)으로 갈라졌다.
**`colcon`은 ROS에 종속되지 않은 범용 도구**라서 ROS 1 패키지도, 순수 CMake 패키지도 빌드할 수 있다.

---

# 6. `ament_cmake`와 `ament_python`

```text
ament_cmake          C++ 패키지. CMakeLists.txt 로 빌드
ament_python         순수 Python 패키지. setup.py 로 설치
ament_cmake_python   C++ 과 Python 을 한 패키지에 (인터페이스 + 노드 등)
```

어느 쪽인지는 `package.xml`의 `<build_type>`이 결정한다.

**msg/srv/action을 정의하는 패키지는 `ament_cmake`여야 한다.**
인터페이스 생성이 CMake 매크로로 되어 있어서, Python 노드만 쓸 계획이어도
인터페이스 패키지는 따로 C++ 쪽으로 만드는 것이 관례다.

```text
my_msgs/          ament_cmake     msg/ srv/ 정의만
my_nodes/         ament_python    my_msgs 에 의존하는 Python 노드
```

---

# 7. `ros2 pkg create`

```bash
# C++
ros2 pkg create my_slam \
  --build-type ament_cmake \
  --dependencies rclcpp sensor_msgs tf2_ros

# Python
ros2 pkg create my_tools \
  --build-type ament_python \
  --dependencies rclpy std_msgs

# 노드 파일까지 같이
ros2 pkg create my_slam --build-type ament_cmake \
  --dependencies rclcpp --node-name slam_node
```

`--dependencies`에 적은 것이 `package.xml`과 `CMakeLists.txt`에 **자동으로 들어간다.**
나중에 추가할 때는 두 파일을 직접 고쳐야 한다는 것을 기억해 두자.

반드시 `src/` 아래에서 실행한다.

```bash
cd ~/ros2_ws/src
ros2 pkg create ...
```

---

# 8. C++ 패키지 — `CMakeLists.txt`의 최소 형태

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_slam)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(slam_node src/slam_node.cpp)
ament_target_dependencies(slam_node rclcpp sensor_msgs)

install(TARGETS slam_node
  DESTINATION lib/${PROJECT_NAME})          # ← ros2 run 이 여기를 본다

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})        # ← ros2 launch 가 여기를 본다

ament_package()                             # ← 반드시 마지막
```

세 줄이 특히 중요하다.

```text
install(TARGETS ... DESTINATION lib/${PROJECT_NAME})
    이게 없으면 빌드는 되는데 ros2 run 이 실행 파일을 못 찾는다

install(DIRECTORY launch ... DESTINATION share/${PROJECT_NAME})
    이게 없으면 launch 파일을 못 찾는다

ament_package()
    share/<pkg>/ 에 메타데이터를 깐다. 맨 마지막에 한 번만
```

**"빌드는 성공했는데 `ros2 run`이 패키지를 못 찾는다"의 90%가 `install()` 누락이다.**

---

# 9. Python 패키지 — `setup.py`

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_tools'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),          # ← 이게 있어야 패키지로 인식된다
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'bag_dump = my_tools.bag_dump:main',    # ← ros2 run 이 쓰는 이름
        ],
    },
)
```

C++ 쪽의 `install(TARGETS ...)`에 해당하는 것이 `entry_points`다.

```text
'bag_dump = my_tools.bag_dump:main'
 └실행이름┘   └모듈경로────┘ └함수┘

ros2 run my_tools bag_dump
```

`resource/<package_name>` 빈 파일도 필요하다.
**ament index에 등록되는 표시**라서, 없으면 빌드는 되는데 `ros2 pkg list`에 안 나온다.

---

# 10. `colcon build` — 무엇이 생기나

```text
ros2_ws/
├── src/          내가 쓴 것.        git 에 올린다
├── build/        중간 산출물         gitignore
├── install/      최종 결과물         gitignore
└── log/          빌드 로그           gitignore
```

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

**반드시 workspace 루트에서 실행한다.** `src/` 안에서 하면 거기에 또 `build/`가 생긴다.

Chapter 13에서 본 것처럼 `build/`와 `install/`은 금방 수 GB가 된다.

```gitignore
build/
install/
log/
```

---

# 11. 알아야 할 `colcon` 옵션

```bash
colcon build --packages-select my_slam        # 그것만
colcon build --packages-up-to my_slam         # 그것 + 의존하는 것들
colcon build --packages-above my_msgs         # 그것 + 그것에 의존하는 것들 ★
colcon build --packages-skip-build-finished   # 이미 끝난 건 건너뛴다
```

`--packages-above`가 실무에서 특히 유용하다.

```text
my_msgs 의 .msg 파일을 고쳤다
   → my_msgs 만 다시 빌드하면 그걸 쓰는 노드들은 옛 헤더를 쓴다
   → --packages-above my_msgs 로 의존하는 것들까지 다시 빌드
```

**메시지 정의를 고치고 이상한 런타임 에러가 나면 이걸 의심한다.**

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release   # 최적화 (로봇에 올릴 때)
colcon build --parallel-workers 4                      # Jetson 에서 메모리 부족하면
colcon build --event-handlers console_direct+          # 빌드 출력 그대로 보기
```

Jetson에서 큰 workspace를 빌드하다 멈추면 대개 **메모리 부족**이다.
`--parallel-workers`를 줄이면 해결되는 경우가 많다. (Chapter 12)

빌드에서 제외하려면 그 폴더에 빈 파일 하나를 둔다.

```bash
touch src/some_experiment/COLCON_IGNORE
```

---

# 12. `--symlink-install` — 왜 거의 항상 쓰나

colcon 문서의 설명은 이렇다.

```text
--symlink-install
    Use symlinks instead of copying files from the source
    and build directories where possible.
```

복사 대신 심볼릭 링크를 건다는 뜻인데, 실무적 차이가 크다.

```text
              launch/config/Python 파일을 고쳤을 때

없이          install/ 에 복사본이 있다  →  colcon build 를 다시 해야 반영
--symlink-    install/ 이 src/ 를 가리킨다 →  저장하는 순간 반영
install
```

**Python 노드나 launch 파일을 고칠 때 재빌드가 필요 없어진다.**
C++ 소스는 어차피 컴파일이 필요하므로 해당 없다.

주의: 이미 `--symlink-install` 없이 빌드한 workspace에 나중에 옵션을 붙이면
섞여서 이상하게 동작할 수 있다. 그럴 땐 지우고 다시 한다.

```bash
rm -rf build install log
colcon build --symlink-install
```

---

# 13. `--merge-install`

```text
--merge-install
    Use the --install-base as the install prefix for all packages
    instead of a package specific subdirectory in the install base.
    Without this option each package will contribute its own paths
    to environment variables …
```

기본값은 패키지마다 `install/<pkg>/` 하위 폴더를 만드는 것이다.
패키지가 수백 개가 되면 `AMENT_PREFIX_PATH` 같은 환경변수가 지나치게 길어진다.

```text
기본           install/pkg_a/  install/pkg_b/  …   개발에 편하다
--merge-install  install/ 하나에 전부              배포 이미지에 적합
```

Docker 이미지를 만들 때는 `--merge-install`이 깔끔하다.

---

# 14. `setup.bash`와 `local_setup.bash`

install 공간에는 두 개가 있고, 차이가 중요하다.

```text
local_setup.bash    이 workspace 것만 환경에 추가한다
setup.bash          이 workspace 가 빌드될 때 깔려 있던 underlay 까지
                    같이 source 한다 (chaining)
```

그래서 보통은 이렇게만 하면 된다.

```bash
source ~/ros2_ws/install/setup.bash      # /opt/ros/humble 도 같이 딸려온다
```

```text
source install/setup.bash
        │
        ├── /opt/ros/humble/setup.bash 를 먼저 부른다  (빌드 시점의 underlay)
        └── 그 다음 이 workspace 를 얹는다
```

underlay를 직접 고르고 싶을 때만 `local_setup.bash`를 쓴다.

```bash
source /opt/ros/humble/setup.bash
source ~/other_ws/install/local_setup.bash
source ~/ros2_ws/install/local_setup.bash     # 여러 workspace 를 쌓을 때
```

**나중에 source한 것이 이긴다.** 같은 이름의 패키지가 있으면 마지막 것이 쓰인다.

---

# 15. `rosdep` — 의존성을 한 번에 설치

`package.xml`에 적힌 의존성을 실제 apt 패키지로 바꿔서 설치해준다.

```bash
# 최초 1회
sudo rosdep init
rosdep update

# workspace 의 모든 의존성 설치
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

```text
--from-paths src   src/ 아래 패키지들의 package.xml 을 읽는다
--ignore-src       src 에 이미 있는 것은 apt 로 안 깐다
-r                 일부 실패해도 계속
-y                 확인 안 물어봄
```

**새 컴퓨터나 Docker 이미지를 만들 때 이 한 줄이 의존성 설치를 대신한다.**
`apt install ros-humble-...`을 손으로 나열할 필요가 없어진다.
그래서 `package.xml`을 성실히 채워두는 것이 나중에 값을 한다.

---

# 16. `vcstool` — 여러 저장소를 한 번에

패키지가 여러 git 저장소에 흩어져 있을 때 쓴다.

```yaml
# repos.yaml
repositories:
  livox_ros_driver2:
    type: git
    url: https://github.com/Livox-SDK/livox_ros_driver2.git
    version: master
  my_slam:
    type: git
    url: git@github.com:org/my_slam.git
    version: v0.3.0
```

```bash
sudo apt install python3-vcstool

vcs import src < repos.yaml      # 전부 clone
vcs pull src                     # 전부 pull
vcs status src                   # 전부 상태 확인
vcs export src > repos.lock.yaml # 지금 커밋을 고정해서 기록  ★
```

`vcs export`가 유용하다. **Rust의 `Cargo.lock`과 같은 역할**이다 (Rust Chapter 4).
지금 어느 커밋으로 빌드했는지를 파일로 남겨서 나중에 재현할 수 있다.

---

# 17. `bloom` — 바이너리로 배포하기

지금까지는 소스에서 빌드했다. 반대 방향, 즉 `apt install`로 깔 수 있게 만드는 도구가 `bloom`이다.

```text
소스 저장소
   │  bloom
   ▼
debian/ 메타데이터 생성
   │
   ▼
.deb 패키지  →  apt install ros-humble-my-slam
```

사내 apt 저장소를 운영하는 경우가 아니면 직접 쓸 일은 드물다.
다만 **`apt install ros-humble-xxx`로 깔리는 모든 패키지가 이 경로를 거쳤다**는 것은 알아둘 만하다.

```text
binary install (apt)      빠르다. 버전이 배포판에 고정된다
source build (colcon)     느리다. 코드를 고칠 수 있다. 버전 자유
```

---

# 18. 빌드가 깨질 때

```text
증상                                  먼저 볼 것
──────────────────────────────        ────────────────────────────────────
find_package(X) 실패                   그 패키지를 source 했나
                                      rosdep install 로 깔았나
                                      ros2 pkg prefix X

빌드는 됐는데 ros2 run 이 못 찾음        install(TARGETS ... DESTINATION
                                        lib/${PROJECT_NAME}) 이 있나
                                      source install/setup.bash 를 했나

launch 파일을 못 찾음                   install(DIRECTORY launch ...)

Python 패키지가 ros2 pkg list 에 없음    resource/<pkg_name> 파일이 있나
                                      setup.py 의 data_files 확인

msg 를 고쳤는데 옛 정의로 동작           --packages-above <msg_pkg> 로 재빌드

빌드 중 멈춤 / OOM                      --parallel-workers 를 줄인다
                                      free -h 로 메모리 확인

이유 없이 계속 이상함                    rm -rf build install log 후 재빌드
```

마지막 줄은 농담이 아니다. **`build/`에 남은 옛 산출물이 원인인 경우가 실제로 많다.**
특히 브랜치를 바꾸거나 패키지 이름을 바꿨을 때 그렇다.

---

# 19. Docker와 함께

Chapter 9.5의 구분이 그대로 적용된다.

```text
이미지에 굽는다                      마운트한다 (/ws)
────────────────────────           ─────────────────────
ROS 2 배포판 (/opt/ros/humble)       내 src/
apt 로 깔리는 드라이버 패키지          build/ install/ log/
rosdep 으로 설치한 의존성
```

Dockerfile에서 의존성만 먼저 설치하면 레이어 캐시가 산다.

```dockerfile
# package.xml 만 먼저 복사해서 의존성 설치 → 소스가 바뀌어도 이 레이어는 재사용
COPY src/*/package.xml /tmp/pkgs/
RUN rosdep install --from-paths /tmp/pkgs --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*
```

그리고 **빌드는 한쪽에서만 한다.** 호스트에서 만든 `install/`을 컨테이너가 쓰면
배포판과 ABI가 달라 깨진다. (Chapter 9.5 §27)

---

# 20. Mini Practice

```bash
# 1) workspace 와 패키지 만들기
mkdir -p ~/test_ws/src && cd ~/test_ws/src
ros2 pkg create hello_cpp --build-type ament_cmake --dependencies rclcpp --node-name hello_node
ros2 pkg create hello_py  --build-type ament_python --dependencies rclpy

# 2) 무엇이 만들어졌나 비교
ls -R hello_cpp hello_py
cat hello_cpp/package.xml | grep build_type
cat hello_py/package.xml  | grep build_type

# 3) 빌드하고 어디에 설치됐는지
cd ~/test_ws
colcon build --symlink-install
find install -name "hello_node"
ls install/hello_cpp/share/hello_cpp/

# 4) source 전후 비교
ros2 pkg list | grep hello        # 아무것도 안 나온다
source install/setup.bash
ros2 pkg list | grep hello        # 이제 나온다
ros2 pkg prefix hello_cpp

# 5) install() 을 지우면?
#    hello_cpp/CMakeLists.txt 의 install(TARGETS ...) 를 주석 처리하고
colcon build --packages-select hello_cpp
ros2 run hello_cpp hello_node     # 빌드는 됐는데 못 찾는다

# 6) setup.bash vs local_setup.bash
env | grep AMENT_PREFIX_PATH
# 새 shell 에서
source install/local_setup.bash
ros2 run hello_cpp hello_node     # /opt/ros 가 없어서 실패할 수 있다
```

5번과 6번이 이 chapter의 핵심이다.
**빌드 성공 ≠ 실행 가능**이고, **`install()`과 `source`가 그 사이를 잇는다.**

---

# 21. 오늘의 핵심

```text
   src/                    package.xml  ← 무엇에 의존하나 (colcon, rosdep 이 읽는다)
    │                      CMakeLists   ← C++ 을 어떻게 빌드/설치하나
    │                      setup.py     ← Python 을 어떻게 설치하나
    │
    │   colcon build --symlink-install
    │        build tool 이 순서를 정하고
    │        build system(ament) 이 각각을 빌드
    ▼
   build/                  중간 산출물
   install/                lib/<pkg>/    ← ros2 run 이 보는 곳
    │                      share/<pkg>/  ← ros2 launch, find_package 가 보는 곳
    │
    │   source install/setup.bash
    ▼
   현재 shell 이 패키지를 찾을 수 있게 된다
```

---

# 22. 반드시 구분할 것

```text
Build system  ≠  Build tool
   ament(패키지 하나)  vs  colcon(여러 개의 순서)

ament_cmake  ≠  ament_python
   msg/srv 를 정의하려면 ament_cmake 여야 한다

package.xml  ≠  CMakeLists.txt
   전자는 의존성 선언(도구들이 읽는다), 후자는 빌드 지시

빌드 성공  ≠  ros2 run 가능
   install() 과 source 가 사이에 있다

setup.bash  ≠  local_setup.bash
   전자는 underlay 까지 딸려온다

--packages-select  ≠  --packages-up-to  ≠  --packages-above
   그것만 / 그것이 의존하는 것까지 / 그것에 의존하는 것까지

/opt/ros/humble  ≠  ~/ros2_ws/install
   배포판(underlay)  vs  내가 빌드한 것(overlay)

apt install  ≠  colcon build
   바이너리 설치 vs 소스 빌드
```

---

# 23. Chapter 연결

```text
Chapter 6
ROS 2 — node, topic, DDS, QoS (무엇인지)

Chapter 6.2  ← 여기
파일 시스템과 빌드 (어디에 놓이고 어떻게 만들어지는지)

Chapter 6.5
micro-ROS — 이 stack 을 MCU 까지 내리면

Chapter 9.5
Docker — /opt/ros 는 이미지에, install/ 은 마운트에

Chapter 12
빌드 중 OOM 과 --parallel-workers

Chapter 13
build/ install/ 이 디스크를 먹는다

Rust Chapter 1~4
workspace / package / 의존성 / lock 파일 — 같은 문제를 다른 언어가 푼 방식
```

**출처:** `--symlink-install`과 `--merge-install`의 설명은
[colcon build 레퍼런스](https://colcon.readthedocs.io/en/released/reference/verb/build.html)의 원문이다.
