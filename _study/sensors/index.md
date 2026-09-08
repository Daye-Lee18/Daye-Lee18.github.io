---
layout: study-topic
title: Sensors & I/O
description: 센서마다 다른 물리 인터페이스 · 설정 프로토콜 · 데이터 프로토콜 · ROS 토픽을 한 형식으로 정리한다.
permalink: /study/sensors/
topic_index: true
---

센서를 하나 붙일 때마다 매번 같은 것을 찾아 헤매게 된다.
IP가 뭐였지, 설정은 웹으로 하나 SDK로 하나, 데이터는 몇 번 포트로 오나,
드라이버는 뭘 깔아야 하나, 토픽 이름이 뭐였나.

이 섹션은 그걸 **센서마다 같은 7줄로 기록**해 두는 곳이다.

```text
1. 물리 인터페이스    Ethernet / USB / CAN / Serial
2. 설정 프로토콜      HTTP CGI / TCP / SDK / JSON 파일
3. 데이터 프로토콜    UDP / TCP / USB stream
4. 포트 · 주소        기본값과 우리 장비의 실제 값
5. ROS 드라이버       패키지 이름과 저장소
6. raw 토픽           드라이버가 처음 뱉는 것
7. 출력 토픽          실제로 쓰는 PointCloud2 / Image / Imu
```

핵심은 **설정용 통신과 데이터용 통신이 대개 분리되어 있다**는 것이다.

```text
    Thor  ──── HTTP / TCP ────▶  LiDAR       설정·조회
    Thor  ◀─── UDP ────────────  LiDAR       실제 점군 데이터
```

`ping`이 되는데 점군이 안 온다면 대개 이 두 경로 중 하나만 열려 있는 것이다.
