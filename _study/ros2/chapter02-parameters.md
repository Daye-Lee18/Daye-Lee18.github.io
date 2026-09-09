---
title: "Chapter 2. 파라미터 — 값을 어디에 적고 누가 읽나"
importance: 3
---

> **Goal:** 파라미터가 노드에 도달하는 세 가지 경로를 구분하고,
> YAML 파일의 노드 이름이 왜 실제 노드 이름과 같아야 하는지 안다.

Chapter 1에서 `--params-file`과 `-p`를 봤다. 그런데 값을 넣었는데 안 먹는 일이 자주 생긴다.
원인은 대개 **YAML의 노드 이름**이거나 **선언(declare)을 안 한 것**이다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `ros2 param list` · `ros2 param get` · `ros2 param dump`

| 명령어                                                 | 하는 일                             |
| :----------------------------------------------------- | :---------------------------------- |
| `ros2 param list`                                      | 전체 노드의 파라미터                |
| `ros2 param list /my_node`                             | 그 노드의 파라미터만                |
| `ros2 param get /my_node max_range`                    | 현재 값                             |
| `ros2 param set /my_node max_range 80.0`               | **실행 중에** 값 바꾸기             |
| `ros2 param dump /my_node`                             | 현재 값을 YAML로 출력               |
| `ros2 param describe /my_node max_range`               | 타입·설명·범위                      |
| `ros2 node info /my_node`                              | 그 노드 전체 (토픽·서비스·파라미터) |
| `ros2 run <pkg> <exe> --ros-args -p k:=v`              | 실행할 때 하나                      |
| `ros2 run <pkg> <exe> --ros-args --params-file f.yaml` | 실행할 때 파일로                    |

---

# 1. 값이 노드에 도달하는 세 경로

```text
① 코드의 기본값        declare_parameter("max_range", 100.0)
                      아무것도 안 주면 이 값

② 명령줄 / launch      --ros-args -p max_range:=80.0
                      또는 --params-file, 또는 launch 의 parameters=[]
                      → ①을 덮어쓴다

③ 실행 중 변경         ros2 param set /my_node max_range 60.0
                      → 지금 돌고 있는 노드의 값만. 파일에는 안 남는다
```

**③은 껐다 켜면 사라진다.** 튜닝할 때 편하지만, 좋은 값을 찾았으면 YAML에 옮겨 적어야 한다.

```bash
ros2 param set /slam_estimator max_range 60.0    # 돌려보고
ros2 param dump /slam_estimator > tuned.yaml     # 좋으면 뽑아서
#  → config/slam.yaml 에 반영
```

`ros2 param dump`가 그 다리 역할을 한다.

---

# 2. YAML 파일의 구조

여기서 가장 많이 틀린다. 파라미터 파일은 **평평한 key-value가 아니다.**

```yaml
# config/slam.yaml
slam_estimator: # ← 노드 이름. 실제 노드 이름과 같아야 한다
  ros__parameters: # ← 이 키가 반드시 있어야 한다 (밑줄 두 개)
    max_range: 100.0
    min_range: 0.5
    use_imu: true
    frame_id: "base_link"
    extrinsic_T: [0.0, 0.0, 0.1]
```

두 층이 필수다.

```text
노드 이름
  ros__parameters:        ← ros 뒤에 밑줄 2개. ros_parameters 가 아니다
    실제 값들
```

**노드 이름이 안 맞으면 조용히 무시된다.** 에러도 안 난다.

```text
YAML 에 적힌 이름     slam_estimator
실제 노드 이름        slam_node          ← 안 맞는다
결과                 파일은 읽히는데 값이 적용되지 않는다
```

실제 노드 이름 확인:

```bash
ros2 node list
```

노드 이름을 모르거나 여러 개에 같은 값을 주고 싶으면 와일드카드를 쓴다.

```yaml
/**: # 모든 노드
  ros__parameters:
    use_sim_time: true
```

`use_sim_time`처럼 전부에 걸어야 하는 값에 특히 쓸모 있다.

네임스페이스가 있으면 경로까지 맞춰야 한다.

```yaml
/front/slam_estimator: # namespace 가 front 인 경우
  ros__parameters:
    max_range: 100.0
```

---

# 3. 선언하지 않은 파라미터는 무시된다

ROS 2는 노드가 **미리 선언한** 파라미터만 받는다.

```cpp
// 선언이 있어야 한다
this->declare_parameter<double>("max_range", 100.0);
double r = this->get_parameter("max_range").as_double();
```

```python
self.declare_parameter('max_range', 100.0)
r = self.get_parameter('max_range').value
```

선언 안 된 이름을 YAML에 적으면 **그냥 버려진다.**

```text
YAML 에 max_rnage: 80.0   ← 오타
노드는 max_range 를 선언
결과                      오타 난 쪽은 무시되고, max_range 는 기본값 100.0 으로 남는다
                          에러 없음. 조용히 틀린 값으로 돈다
```

Chapter 9의 CSV 열 개수 이야기와 같은 종류의 함정이다 —
**동작은 하는데 값이 틀린다.** 확인 습관이 필요하다.

```bash
ros2 param get /slam_estimator max_range     # 진짜 80.0 인가?
```

선언 없이도 받게 하려면 옵션을 켠다. 다만 오타를 못 잡게 되므로 권장되지 않는다.

```cpp
rclcpp::NodeOptions().allow_undeclared_parameters(true);
```

---

# 4. `run`으로 넘기기

```bash
# 하나
ros2 run slam_ros2 estimator --ros-args -p max_range:=80.0

# 타입에 주의 — 100 은 int, 100.0 은 double
ros2 run slam_ros2 estimator --ros-args -p max_range:=100.0

# 문자열
ros2 run slam_ros2 estimator --ros-args -p frame_id:=base_link

# 배열
ros2 run slam_ros2 estimator --ros-args -p "extrinsic_T:=[0.0, 0.0, 0.1]"

# 파일
ros2 run slam_ros2 estimator --ros-args --params-file config/slam.yaml

# 파일 + 개별 덮어쓰기 (뒤가 이긴다)
ros2 run slam_ros2 estimator --ros-args \
  --params-file config/slam.yaml \
  -p max_range:=50.0
```

**타입이 엄격하다.** `double`로 선언한 것에 `100`을 주면 int로 해석되어 거부된다.

```text
declare_parameter<double>("max_range", 100.0)
-p max_range:=100      →  타입 불일치 에러
-p max_range:=100.0    →  OK
```

---

# 5. `launch`에서 넘기기

```python
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

cfg = os.path.join(get_package_share_directory('slam_ros2'), 'config', 'slam.yaml')

Node(
    package='slam_ros2',
    executable='estimator',
    name='slam_estimator',        # ← YAML 의 노드 이름과 이게 같아야 한다
    parameters=[
        cfg,                       # 파일
        {'use_sim_time': True},    # 개별 값 (뒤가 이긴다)
    ],
)
```

`name=`이 열쇠다. **launch에서 노드 이름을 정하고, YAML은 그 이름을 찾는다.**
둘이 어긋나면 §2의 조용한 실패가 난다.

launch 인자로 밖에서 받을 수도 있다.

```python
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

DeclareLaunchArgument('max_range', default_value='100.0'),

Node(
    ...,
    parameters=[{'max_range': LaunchConfiguration('max_range')}],
)
```

```bash
ros2 launch slam_ros2 bringup.launch.py max_range:=50.0
ros2 launch -s slam_ros2 bringup.launch.py     # 받을 수 있는 인자 보기
```

---

# 6. 파일이 어디 있어야 하나

Chapter 1 §4와 같다. **`install/` 쪽을 읽는다.**

```cmake
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})
```

```python
# launch 안에서는 항상 이렇게
get_package_share_directory('slam_ros2')     # install/slam_ros2/share/slam_ros2
```

상대 경로 `config/slam.yaml`을 쓰면 **어느 디렉토리에서 실행했느냐에 따라** 깨진다.
`ros2 run`으로 테스트할 때는 되는데 `ros2 launch`에서 안 되는 전형적인 이유다.

`--symlink-install`이면 YAML을 고쳐도 재빌드가 필요 없다. (Chapter 1 §5)

---

# 7. 실행 중 바꾸기

```bash
ros2 param list /slam_estimator
ros2 param get  /slam_estimator max_range
ros2 param set  /slam_estimator max_range 60.0
```

다만 **노드가 그 변경을 실제로 반영하느냐는 별개**다.

```text
노드가 매 루프마다 get_parameter() 를 호출한다   → 바로 반영
초기화할 때 한 번만 읽어서 멤버에 저장했다        → 값은 바뀌는데 동작은 그대로
```

후자가 흔하다. `ros2 param get`은 60.0을 보여주는데 동작은 100.0 그대로인 상태.
콜백을 등록해 둔 노드라면 반영된다.

```cpp
// 값이 바뀔 때 알림을 받는다
auto cb = this->add_on_set_parameters_callback(...);
```

**튜닝이 안 먹는 것 같으면 이걸 의심한다.** 노드를 재시작하면 확실하다.

---

# 8. 안 될 때

```text
증상                                  먼저 볼 것
──────────────────────────────        ────────────────────────────────
값이 적용 안 됨 (에러도 없음)           YAML 의 노드 이름 == ros2 node list 결과인가
                                      ros__parameters (밑줄 2개) 가 있나
                                      노드가 declare_parameter 했나
                                      → ros2 param get 으로 실제 값 확인

타입 에러                              100 vs 100.0 (int vs double)
                                      선언 타입과 맞는가

파일을 못 찾음                          get_package_share_directory 를 썼나
                                      install(DIRECTORY config ...) 이 있나

param set 은 되는데 동작이 그대로        노드가 값을 한 번만 읽는다 (§7)
                                      재시작해서 확인

launch 에서만 안 됨                     name= 과 YAML 노드 이름이 다르다
                                      상대 경로를 썼다
```

---

# 9. Mini Practice

```bash
# 1) 돌고 있는 노드의 파라미터 구경
ros2 run demo_nodes_cpp talker &
ros2 param list
ros2 param list /talker
ros2 param get /talker use_sim_time

# 2) YAML 을 만들어 넘겨본다
cat > /tmp/p.yaml <<'YAML'
talker:
  ros__parameters:
    use_sim_time: false
YAML
ros2 run demo_nodes_cpp talker --ros-args --params-file /tmp/p.yaml

# 3) 노드 이름을 일부러 틀려본다
cat > /tmp/bad.yaml <<'YAML'
wrong_name:
  ros__parameters:
    use_sim_time: true
YAML
ros2 run demo_nodes_cpp talker --ros-args --params-file /tmp/bad.yaml
#    다른 터미널에서
ros2 param get /talker use_sim_time     # 적용됐나? 에러는 났나?

# 4) 현재 값을 파일로 뽑기
ros2 param dump /talker

# 5) 실행 중 변경
ros2 param set /talker use_sim_time true
ros2 param get /talker use_sim_time
```

3번이 핵심이다. **틀린 노드 이름은 에러 없이 무시된다.**
이걸 한 번 눈으로 보면 나중에 "값이 안 먹는" 상황에서 바로 여기를 의심하게 된다.

---

# 10. 오늘의 핵심

```text
   코드         declare_parameter("max_range", 100.0)    ← 선언 없으면 무시된다
     │
     │  덮어쓴다
     ▼
   YAML        slam_estimator:              ← 실제 노드 이름과 같아야 한다
                 ros__parameters:           ← 밑줄 2개
                   max_range: 80.0
     │
     │  덮어쓴다
     ▼
   -p k:=v     명령줄 / launch 의 개별 값
     │
     │  덮어쓴다 (실행 중, 휘발성)
     ▼
   ros2 param set        → dump 로 뽑아서 YAML 에 되돌린다

   확인은 항상   ros2 param get <node> <name>
```

---

# 11. 반드시 구분할 것

```text
ros__parameters  ≠  ros_parameters
   밑줄 2개

YAML 의 노드 이름  ≠  패키지 이름
   ros2 node list 에 나오는 그 이름

선언 안 된 파라미터
   조용히 무시된다. 에러가 안 난다

100  ≠  100.0
   int vs double. 선언 타입과 맞아야 한다

param set  ≠  영구 저장
   껐다 켜면 사라진다. dump 로 뽑아 둔다

param get 이 새 값  ≠  동작이 바뀜
   노드가 한 번만 읽었을 수 있다

상대 경로  ≠  get_package_share_directory()
   launch 에서는 후자를 쓴다
```

---

# 12. Chapter 연결

```text
Chapter 1
run 과 launch — --ros-args 로 넘기는 법

Chapter 2  ← 여기
파라미터 — 어디에 적고 누가 읽나

Chapter 3
remapping 과 namespace — 이름이 정해지는 규칙

Edge Computing Chapter 6.2
install(DIRECTORY config ...) — 파일이 install/ 로 가는 경로
```
