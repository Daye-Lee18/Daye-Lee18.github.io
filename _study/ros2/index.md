---
layout: study-topic
title: ROS 2
description: 노드를 실행하고 설정하고 디버깅하는 법 — run/launch, 파라미터, remapping, 네임스페이스.
permalink: /study/ros2/
topic_index: true
---

명령어를 외우는 게 아니라 **어디에 무엇이 설정되어 있는지**를 아는 것이 목표다.
`ros2 run`이 왜 인자를 두 개 받는지, 그 실행 파일 이름은 누가 정했는지,
파라미터는 어디에 적고 누가 읽는지 같은 것들.

```text
Ch 1   ros2 run 과 ros2 launch      노드를 실행하는 두 가지 방법
Ch 2   파라미터                      값을 어디에 적고 누가 읽나
Ch 3   remapping 과 namespace       토픽 이름이 정해지는 규칙
```

## 다른 곳에 있는 ROS 2 내용

지금은 Edge Computing 아래에 흩어져 있다. 겹쳐 쓰지 않고 링크로 연결한다.

```text
Edge Computing Ch 6      ROS 2 개념 — node, topic, service, action,
                         rclcpp/rcl/rmw/DDS, QoS, discovery, executor

Edge Computing Ch 6.2    파일 시스템과 빌드 — package.xml, CMakeLists,
                         setup.py, colcon, rosdep, underlay/overlay

Edge Computing Ch 6.5    micro-ROS — MCU 위의 ROS 2

Edge Computing Ch 9.5    Docker 안에서 ROS 2 — ROS_DOMAIN_ID,
                         RMW_IMPLEMENTATION, CycloneDDS, GUI
```
