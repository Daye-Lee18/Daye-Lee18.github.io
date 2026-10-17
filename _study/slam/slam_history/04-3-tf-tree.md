---
layout: study-chapter
title: "4.3 ROS 2 TF tree for SLAM"
description: "TF의 의미와 map → odom → base_link → lidar/camera 구조, 그리고 TF 조회 주기가 pose에 미치는 영향."
importance: 4.3
category: SLAM
series: slam_history
permalink: /study/slam/history/04-3-tf-tree/
---

## 1. TF란?

**TF = Transform(좌표 변환)**

TF는 ROS 2에서 서로 다른 좌표계 사이의 **위치와 회전 관계**를 관리하는 시스템이다.

예를 들어 로봇이 움직이면 지도에서 로봇의 위치가 계속 바뀐다. TF를 이용하면 지도 좌표계와 로봇 좌표계의 관계를 조회해 그 위치를 알 수 있다.

## 2. `map → base_link`란?

- `map`: 기준점이 고정된 지도 좌표계
- `base_link`: 로봇 몸체에 붙어 로봇과 함께 움직이는 좌표계

`map → base_link`는 **지도 원점에서 로봇까지의 위치와 회전 관계**를 뜻한다.

예를 들어 TF에서 다음 값을 얻었다고 해보자.

```text
map → base_link

x = 5 m
y = 2 m
yaw = 30°
```

이는 지도 원점에서 로봇이 x 방향으로 5 m, y 방향으로 2 m 떨어져 있고, 30° 회전해 있다는 의미다.

## 3. ROS 2에서 자주 보는 좌표계

| 좌표계 | 의미 |
| --- | --- |
| `map` | 전역 지도 기준 |
| `odom` | 로봇의 연속적인 이동 추정 기준 |
| `base_link` | 로봇 몸체 기준 |
| `lidar` | LiDAR 센서 기준 |
| `camera_link` | 카메라 기준 |

보통 다음과 같은 관계로 연결된다.

```text
map
 └── odom
      └── base_link
           ├── lidar
           └── camera_link
```

이를 **TF Tree(좌표계 트리)**라고 부른다.

여기서 `map → odom`은 SLAM의 전역 보정을, `odom → base_link`는 로봇의 연속적인 이동 추정을 나타낸다. `base_link → lidar/camera_link`는 로봇 몸체에 센서가 장착된 위치와 방향이다.

따라서 SLAM 시스템에서는 고립된 `map → base_link` 값 하나만 보기보다 다음 전체 chain을 확인하는 것이 중요하다.

```text
map → odom → base_link → lidar/camera
```

## 4. 코드에서 0.5초가 의미하는 것

다음 코드는 0.5초마다 callback을 실행한다.

```python
self.create_timer(0.5, self.on_tick)
```

`on_tick`에서 TF를 조회해 로봇 위치를 `self.pose`에 저장한다면, `self.pose`는 0.5초마다 갱신된다.

그런데 사진을 찍는 순간 TF를 다시 조회하지 않고 마지막으로 저장한 `self.pose`를 사용한다면, 사진에 붙는 위치는 타이머가 정상적으로 실행된다고 가정했을 때 **최대 약 0.5초 전의 위치**일 수 있다.

예를 들어 로봇 속도가 1 m/s라면:

$$
1\ \mathrm{m/s} \times 0.5\ \mathrm{s} = 0.5\ \mathrm{m}
$$

따라서 사진 촬영 시점과 저장된 pose 사이에 최대 약 0.5 m의 위치 차이가 생길 수 있다.

```text
TF 조회 시점 ───── 최대 0.5초 ───── 사진 촬영 시점
     pose A                              실제 pose B
```

이 문제를 줄이려면 사진 callback에서 사진의 timestamp에 맞는 TF를 조회해야 한다. 단순히 가장 최근에 저장한 pose를 재사용하면 로봇이 빠르게 움직일수록 위치 오차가 커진다.

---

> **핵심:** TF는 좌표계 간의 관계를 알려주는 시스템이고, `map → base_link`는 지도 기준으로 로봇이 어디에 있고 어느 방향을 보고 있는지를 나타내는 변환이다. 실제 SLAM에서는 `map → odom → base_link → lidar/camera` 전체 TF tree와 각 데이터의 timestamp를 함께 확인해야 한다.
