---
title: "Chapter 13. Storage & Data Logging"
importance: 14
---

> **Goal:** 로봇이 하루에 몇 GB를 만들어내는지 직접 계산하고, 그 데이터를
> **어디에·얼마나·어떤 방식으로** 남길지 결정할 수 있다.
> 디스크가 꽉 차거나 저장장치가 수명을 다했을 때 무슨 일이 벌어지는지도 이해한다.

Chapter 12에서 전력과 발열이 예산이라는 것을 봤다.
저장장치도 똑같이 예산이다. 다만 두 가지가 동시에 닳는다 — **용량**과 **쓰기 수명**이다.
그리고 이 둘은 조용히 닳다가, 하필 현장에서 한꺼번에 터진다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `df -h` · `du -sh *` · `ros2 bag record`

| 명령어                               | 하는 일                                            |
| :----------------------------------- | :------------------------------------------------- |
| `df -h`                              | 파티션별 남은 용량                                 |
| `df -i`                              | **inode** 잔량 (용량은 남았는데 못 쓸 때)          |
| `du -sh *`                           | 현재 디렉토리에서 뭐가 용량을 먹는지               |
| `du -sh /var/log/* \| sort -h`       | 로그 중 큰 것부터                                  |
| `lsblk -f`                           | 디스크·파티션·파일시스템 구조                      |
| `sudo lsof +L1`                      | **지운 줄 알았는데 안 줄어들 때** (열려 있는 파일) |
| `iostat -x 1`                        | 디스크 I/O 부하 (sysstat)                          |
| `sudo iotop -o`                      | 어느 프로세스가 쓰고 있는지                        |
| `journalctl --disk-usage`            | 시스템 로그가 차지한 용량                          |
| `sudo journalctl --vacuum-size=200M` | 시스템 로그 줄이기                                 |
| `docker system df`                   | Docker가 먹은 용량                                 |
| `docker system prune -a`             | 안 쓰는 이미지·컨테이너 정리                       |
| `ros2 bag record -a -o run1`         | 전체 토픽 기록                                     |
| `ros2 bag record /imu /odom -o run1` | **선택** 기록                                      |
| `ros2 bag info run1`                 | 기록된 토픽·크기·기간                              |
| `sync`                               | 버퍼를 디스크에 강제로 내림 (전원 끄기 전)         |

---

# 1. 로봇은 데이터를 만들어내는 기계다

로봇 이야기를 할 때 저장장치는 보통 마지막에 나온다. 그런데 숫자를 한 번 내보면 순서가 바뀐다.

센서 하나하나의 초당 데이터량을 계산해 보자.
LiDAR는 `점 개수 × 점 하나의 크기`, 카메라는 `가로 × 세로 × 채널 × fps`다.

```text
센서                                     MB/s      GB/시간
────────────────────────────────────    ──────    ────────
VLP-16 LiDAR (16ch × 1800 × 10Hz)          4.6       16.6
Ouster OS1-64 (64ch × 1024 × 10Hz)        10.5       37.7
1080p30 카메라, 무압축 RGB8              186.6      671.8
1080p30 카메라, JPEG 약 10:1              18.7       67.2
720p30 카메라, JPEG 약 10:1                8.3       29.9
IMU 200Hz                                  0.1        0.2
joint state 500Hz                          0.5        1.8
```

VLP-16 계산을 손으로 따라가 보면 이렇다.

```text
16 채널 × 1800 방위각 × 10 Hz          = 288,000 점/초
점 하나 = XYZI float32 4개             = 16 바이트
288,000 × 16                           = 4.6 MB/s
4.6 MB/s × 3600                        = 16.6 GB/시간
```

여기서 눈여겨볼 것은 **무압축 1080p 카메라 한 대가 LiDAR의 40배**라는 점이다.
로봇 저장장치를 잡아먹는 것은 거의 항상 영상이다.

---

# 2. 전부 기록하면 하루에 얼마인가

LiDAR 1대 + 카메라 1대(JPEG) + IMU + joint state를 전부 기록한다고 하자.

```text
합계  23.8 MB/s  =  85.8 GB/시간

하루 8시간 운용   →  687 GB/일
```

이제 저장장치가 얼마나 버티는지 나눠 보자.

```text
저장장치        가득 차기까지
──────────    ──────────────
 32 GB            0.4 시간   ← 22분
 64 GB            0.7 시간   ← 45분
256 GB            3.0 시간
  1 TB           11.6 시간
```

**64 GB microSD는 45분이면 꽉 찬다.** 점심 먹고 오면 로봇이 멈춰 있다.

---

# 3. 저장장치의 종류

|           | microSD        | eMMC            | NVMe SSD          | USB 저장장치  |
| :-------- | :------------- | :-------------- | :---------------- | :------------ |
| 위치      | 슬롯에 꽂음    | 보드에 납땜     | M.2 슬롯          | 외부 포트     |
| 속도      | ~90 MB/s       | ~250 MB/s       | 1,000~3,000 MB/s  | 포트에 따라   |
| 쓰기 수명 | **매우 짧음**  | 중간            | 김                | 짧음          |
| 진동      | 접촉 불량 위험 | **강함** (납땜) | 강함              | 빠질 위험     |
| 로봇에서  | 개발·부팅용    | rootfs          | **데이터 기록용** | 데이터 회수용 |

Jetson 개발 키트는 microSD로 부팅하는 구성이 많지만,
**실제 로봇에 올릴 때는 NVMe를 다는 것이 사실상 기본**이다. 이유가 다음 절이다.

---

# 4. 용량보다 먼저 닳는 것 — 쓰기 수명

플래시 메모리는 같은 자리에 무한히 쓸 수 없다.
셀마다 **P/E cycle(지우고 쓰기)** 횟수가 정해져 있고, 그걸 다 쓰면 그 셀은 죽는다.

앞의 687 GB/일을 64 GB 카드에 쓴다고 하면:

```text
687 GB/일 ÷ 64 GB  =  하루에 카드 전체를 10.7번 덮어씀

P/E cycle    수명
──────────   ──────────
   500회      약 1.5개월
 1,000회      약 3개월
 3,000회      약 10개월      ← 고내구성 카드
```

쓰기 증폭(write amplification)을 무시한 낙관적인 계산이 이 정도다.
실제로는 더 짧다.

NVMe SSD는 TBW(Terabytes Written)로 수명을 표기한다.

```text
500 GB NVMe, TBW 300 TB   →  약 1.2년
  1 TB NVMe, TBW 600 TB   →  약 2.4년
```

한 자릿수 배가 아니라 **10배 이상 차이**다.
"microSD가 자꾸 죽는다"는 문제는 대부분 카드 품질이 아니라 **기록량 설계의 문제**다.

---

# 5. microSD가 로봇에서 특히 잘 죽는 이유

```text
① 쓰기량이 애초에 많다        위 계산대로
② 컨트롤러가 단순하다          웨어 레벨링이 SSD만큼 똑똑하지 않다
③ 진동                       슬롯 접촉 불량
④ 갑작스러운 전원 차단         쓰는 도중 전원이 끊기면 블록이 깨진다
⑤ 조용히 죽는다               읽기는 되는데 쓰기만 실패하는 상태로 한동안 감
```

⑤가 고약하다. 로봇은 잘 도는 것처럼 보이는데 로그만 안 쌓이는 상태가 며칠 간다.

---

# 6. 그래서 무엇을 기록할 것인가

전부 기록하지 않는 것이 정답이다. 필요한 것만 고르면 이렇게 바뀐다.

```text
                            MB/s     GB/시간   8시간
전부 기록                   23.8      85.8     687 GB
선택 기록                    0.9       3.2      25 GB
  (IMU + joint + odom
   + 720p JPEG 1Hz)
                          ───────────────────
                          27배 차이
```

같은 64 GB 카드가 **40분에서 20시간**으로 늘어난다.

무엇을 남길지는 목적에 따라 갈린다.

```text
사고 원인 분석용     명령·상태·에러·타임스탬프. 용량이 거의 안 든다
알고리즘 튜닝용      해당 알고리즘의 입출력만. 예: SLAM이면 LiDAR + IMU + odom
학습 데이터 수집용   영상 원본이 필요. 이때만 큰 저장장치를 감수한다
```

**"나중에 필요할지 모르니 일단 다 기록"은 저장장치를 죽이는 가장 흔한 이유다.**

---

# 7. `ros2 bag` 기본

ROS 2에서 토픽을 파일로 남기는 도구다.

```bash
# 전체 토픽 — 함부로 쓰지 말 것
ros2 bag record -a -o run1

# 선택 기록 — 실전에서는 이쪽
ros2 bag record /mcu/state/imu /odom /tf -o run1

# 기록된 내용 확인
ros2 bag info run1

# 재생 (실제 센서 없이 알고리즘 디버깅)
ros2 bag play run1
```

`ros2 bag play`가 있기 때문에 **현장에 한 번 나가서 bag을 잘 따오면
사무실에서 같은 상황을 몇 번이고 재현할 수 있다.** 이것이 로깅의 가장 큰 실익이다.

---

# 8. bag 파일 쪼개기와 압축

한 파일이 수백 GB가 되면 옮기지도 열지도 못한다. 잘라서 남긴다.

```bash
ros2 bag record -a \
  --max-bag-size 2000000000 \      # 2 GB 마다 새 파일
  --max-bag-duration 300 \         # 또는 5분 마다
  --compression-mode file \
  --compression-format zstd \
  -o run1
```

압축은 공짜가 아니다.

```text
             용량        CPU
─────────    ────────    ──────────────────
무압축        100%        거의 안 씀
zstd         30~50%      코어 하나를 상시 먹을 수 있다
```

Jetson처럼 CPU 여유가 빠듯한 곳에서는
**압축 때문에 제어 루프가 밀리지 않는지** 먼저 확인해야 한다.

---

# 9. 링 버퍼 — 최근 N분만 유지하기

대부분의 경우 알고 싶은 것은 "사고 직전 몇 분"이다. 그 앞의 8시간은 필요 없다.

```text
전부 기록(23.8 MB/s) 기준 링 버퍼 크기

최근  5분   →   7.2 GB
최근 10분   →  14.3 GB
최근 30분   →  42.9 GB
```

30분치를 항상 들고 있어도 43 GB면 된다. 687 GB와 비교하면 16분의 1이다.

구현은 `--max-bag-duration`으로 잘게 쪼개 두고,
오래된 파일을 지우는 간단한 스크립트를 붙이는 식이다.

```bash
# 30분보다 오래된 bag 파일 삭제 (cron 또는 타이머로)
find /data/bags -name '*.mcap' -mmin +30 -delete
```

---

# 10. 사고가 났을 때만 남기기

한 걸음 더 나가면, 평소에는 RAM에만 들고 있다가
**이상이 감지된 순간에만 디스크에 떨군다.**

```text
평소       링 버퍼 (tmpfs = RAM)     ← 디스크에 안 쓴다. 수명 소모 0
   │
   │  넘어짐 감지 / e-stop / 추정 실패
   ▼
이벤트      직전 60초를 디스크에 저장
```

디스크 쓰기 수명을 거의 쓰지 않으면서 필요한 순간은 다 잡는 방식이다.
Chapter 19의 fault detection과 그대로 이어진다.

---

# 11. 디스크가 꽉 차면 무슨 일이 생기나

단순히 "기록이 안 되는" 정도로 끝나지 않는다.

```text
디스크 100%
   │
   ├─ 로그를 못 씀        → 왜 문제가 생겼는지 알 방법이 사라진다
   ├─ 임시 파일 생성 실패  → 멀쩡하던 프로세스가 죽는다
   ├─ DB·설정 파일 손상   → 쓰다 만 파일이 남는다
   ├─ apt·docker 실패     → 복구 작업조차 안 된다
   └─ 부팅 실패           → 최악
```

특히 **"로그를 못 써서 원인을 못 찾는다"**가 악질이다.
디스크가 찼다는 사실 자체를 로그로 알 수 없다.

그래서 디스크 사용량은 반드시 **미리** 감시해야 한다. (Chapter 18)

```bash
df -h /            # 남은 용량
df -i /            # inode
```

---

# 12. `df`와 `du`가 다를 때

가장 흔한 미스터리다.

```text
$ du -sh /var/log
  1.2G  /var/log          ← 파일은 1.2 GB 뿐인데

$ df -h /
  /dev/nvme0n1p1  98% 사용   ← 디스크는 꽉 찼다고?
```

파일을 `rm`으로 지웠지만 **그 파일을 열고 있는 프로세스가 아직 살아 있으면**
공간이 반환되지 않는다. 디렉토리에서 이름만 사라졌을 뿐이다.

```bash
sudo lsof +L1        # 삭제됐지만 열려 있는 파일
```

해법은 그 프로세스를 재시작하는 것이다.

```bash
sudo systemctl restart my_logger
```

`du`는 **파일을 세고**, `df`는 **블록을 센다.** 둘이 다르면 대개 이 경우다.

---

# 13. inode 고갈

용량은 남았는데 `No space left on device`가 뜨는 또 다른 경우다.

```bash
$ df -h /
  /dev/nvme0n1p1  45% 사용     ← 용량은 널널한데

$ df -i /
  /dev/nvme0n1p1  100% 사용    ← inode가 없다
```

파일시스템은 파일 개수도 유한하다. 작은 파일을 수백만 개 만들면
용량보다 inode가 먼저 떨어진다.

```text
잘 생기는 상황
  이미지 프레임을 낱개 파일로 저장  (1초에 30개 × 8시간 = 86만 개)
  로그를 요청마다 새 파일로 저장
```

**작은 파일을 대량으로 만들지 말고 하나의 bag이나 아카이브에 담는 것**이 해법이다.

---

# 14. 시스템 로그 — journald

`journalctl`이 쌓는 로그도 방치하면 몇 GB가 된다.

```bash
journalctl --disk-usage
sudo journalctl --vacuum-size=200M       # 200 MB로 줄이기
sudo journalctl --vacuum-time=7d         # 7일보다 오래된 것 삭제
```

영구 설정은 `/etc/systemd/journald.conf`다.

```ini
[Journal]
SystemMaxUse=500M
SystemMaxFileSize=50M
```

```bash
sudo systemctl restart systemd-journald
```

---

# 15. 애플리케이션 로그 — logrotate

직접 만든 로그 파일은 `logrotate`로 관리한다.

```text
/etc/logrotate.d/my_robot
```

```text
/var/log/my_robot/*.log {
    daily
    rotate 7
    maxsize 100M
    compress
    missingok
    notifempty
    copytruncate
}
```

```text
daily         하루에 한 번 돌린다
rotate 7      7개까지만 보관하고 그 이상은 삭제
maxsize 100M  하루가 안 됐어도 100MB 넘으면 돌린다
compress      돌린 파일은 gzip
copytruncate  프로세스가 파일을 계속 잡고 있어도 안전하게
```

```bash
sudo logrotate -d /etc/logrotate.d/my_robot   # 테스트 (실행 안 함)
sudo logrotate -f /etc/logrotate.d/my_robot   # 강제 실행
```

---

# 16. Docker 로그가 디스크를 채운다

의외로 자주 터지는 경로다. 컨테이너의 stdout은 전부 파일로 쌓인다.

```text
/var/lib/docker/containers/<id>/<id>-json.log
```

**기본 설정에 크기 제한이 없다.** ROS 2 노드가 초당 수십 줄을 찍으면
며칠 만에 수십 GB가 된다.

```bash
docker system df                # Docker가 먹은 용량
docker system prune -a          # 안 쓰는 이미지·컨테이너 정리
```

제한은 `/etc/docker/daemon.json`에 건다.

```json
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "50m", "max-file": "3" }
}
```

```bash
sudo systemctl restart docker
```

컨테이너 하나만 제한하려면 `run` 옵션으로도 된다.

```bash
docker run --log-opt max-size=50m --log-opt max-file=3 ...
```

Chapter 9.5에서 컨테이너를 계속 살려두는 방식으로 바꿨으므로,
**이 설정이 없으면 로그가 무한히 쌓인다.**

---

# 17. 쓰기 부하가 실시간 제어를 방해한다

저장은 용량 문제만이 아니다. **I/O도 자원이다.**

```text
정상                          bag 기록 중
────────────────────         ─────────────────────
제어 루프  2.0 ms 주기        제어 루프  2.0 ~ 9.0 ms
                             ↑ 디스크 flush 순간 튄다
```

microSD처럼 느린 매체에 큰 덩어리를 쓰는 순간,
그 프로세스가 블로킹되면서 CPU와 I/O 큐가 밀린다.

대응은 세 가지다.

```text
① 기록은 별도 디스크로        rootfs와 데이터 디스크를 나눈다
② 기록 프로세스의 우선순위를 낮춤   nice / ionice
③ 제어 루프는 아예 MCU로       Chapter 6.5 — 애초에 같은 칩이 아니다
```

```bash
# 기록 프로세스를 I/O 우선순위 최하로
sudo ionice -c 3 -p <pid>
nice -n 10 ros2 bag record ...
```

---

# 18. 디스크를 나누는 이유

```text
/dev/nvme0n1p1  →  /          rootfs. OS와 프로그램. 거의 안 씀
/dev/nvme0n1p2  →  /data      bag, 로그. 계속 씀
```

이렇게 나누면 **데이터 파티션이 꽉 차도 시스템은 살아 있다.**
로그를 못 남길 뿐, 부팅도 되고 SSH도 되고 지울 수도 있다.

한 파티션에 다 몰아 넣으면 꽉 찬 순간 아무것도 못 한다.

`/etc/fstab`에 등록해 두면 부팅 시 자동으로 마운트된다.

```text
UUID=xxxx-xxxx  /data  ext4  defaults,noatime,nofail  0  2
```

```text
noatime   읽을 때마다 접근시각을 쓰지 않는다 → 쓰기량 감소
nofail    그 디스크가 없어도 부팅은 되게 한다 ← 로봇에서 중요
```

```bash
lsblk -f                    # UUID 확인
sudo mount -a               # fstab 테스트 (재부팅 전에)
```

`nofail`을 빼먹으면 **데이터 디스크를 뽑았을 때 로봇이 부팅되지 않는다.**

---

# 19. tmpfs — RAM에 쓰기

아예 디스크에 안 쓰는 선택지도 있다.

```bash
sudo mount -t tmpfs -o size=2G tmpfs /tmp/ringbuf
```

```text
장점   빠르다. 플래시 수명을 전혀 소모하지 않는다
단점   RAM을 먹는다. 전원이 나가면 사라진다
용도   링 버퍼, 중간 산출물, 자주 덮어쓰는 임시 파일
```

§10의 "사고 순간만 남기기"가 tmpfs 위에서 돌아간다.

---

# 20. 전원이 갑자기 나가면

로봇은 정상 종료 없이 배터리가 빠지는 일이 흔하다.

```text
프로그램이 write() 를 호출
       ↓
커널 페이지 캐시 (RAM)        ← 여기서 전원이 나가면 사라진다
       ↓  (수 초 뒤)
실제 디스크
```

`write()`가 성공했다고 디스크에 있는 게 아니다.

```bash
sync                 # 버퍼를 지금 내린다. 전원 끄기 전 습관
```

중요한 파일은 프로그램에서 `fsync()`를 부르는 것이 정석이고,
ext4의 journaling은 **파일시스템 구조**는 지켜주지만
**쓰다 만 데이터 내용**까지 지켜주지는 않는다.

가장 강한 대책은 rootfs를 읽기 전용으로 두는 것이다.

```text
/       read-only     ← 전원이 나가도 안 깨진다
/data   read-write    ← 깨져도 데이터만 잃는다
```

---

# 21. 데이터를 내리는 방법

기록했으면 가져와야 한다.

```bash
# 네트워크로 (중단돼도 이어받기)
rsync -avz --progress --partial \
  robot@192.168.1.10:/data/bags/ ./bags/

# 큰 파일은 물리적으로 옮기는 게 빠를 때가 많다
```

계산해 보면 답이 나온다.

```text
100 GB 를 옮길 때
  Wi-Fi 5 실측 20 MB/s   →  약 1.4 시간
  1 GbE 실측 100 MB/s    →  약 17 분
  SSD 를 뽑아서 들고 감    →  5 분
```

현장에서는 **NVMe를 통째로 뽑아 오는 것**이 가장 빠른 경우가 많다.

파일 이름에 정보를 넣어 두면 나중에 고생을 덜 한다.

```text
20260908_143022_robot01_stairs_run3.mcap
└날짜──┘ └시각┘ └로봇┘ └상황──┘└회차┘
```

---

# 22. Mini Practice 1

내 로봇이 실제로 초당 몇 바이트를 만드는지 재 본다.

```bash
# 토픽 하나의 실제 대역폭
ros2 topic bw /velodyne_points
ros2 topic bw /camera/image_raw

# 30초만 기록해서 크기를 재고 곱해 본다
timeout 30 ros2 bag record /velodyne_points -o /tmp/test30
du -sh /tmp/test30
```

30초 크기 × 120 = 시간당 GB다. §1의 계산과 맞는지 비교한다.

---

# 23. Mini Practice 2

`df`와 `du`가 어긋나는 상황을 직접 만들어 본다.

```bash
# 큰 파일을 만들고
dd if=/dev/zero of=/tmp/big bs=1M count=1024

# 열어둔 채로 지운다
tail -f /tmp/big &
rm /tmp/big

df -h /tmp          # 공간이 안 줄어든 것을 확인
du -sh /tmp/big     # 파일은 없다고 나온다
sudo lsof +L1       # 범인이 보인다

kill %1             # 프로세스를 죽이면 그제서야 반환된다
df -h /tmp
```

---

# 24. Mini Practice 3

Docker 로그 제한을 확인한다.

```bash
docker system df
sudo du -sh /var/lib/docker/containers/*/*-json.log | sort -h | tail -5
```

가장 큰 것이 수백 MB를 넘으면 `/etc/docker/daemon.json`에 제한을 건다. (§16)

---

# 25. 오늘의 핵심

```text
            저장장치는 두 가지가 동시에 닳는다

   용량                             쓰기 수명
   ────────────────                 ────────────────
   꽉 차면 즉시 터진다               조용히 닳다가 죽는다
   df -h 로 보인다                  거의 안 보인다
   지우면 회복된다                   회복 안 된다

              둘 다 줄이는 방법은 하나

          ┌─────────────────────────┐
          │  필요한 것만, 필요한 만큼  │
          │  선택 기록 + 링 버퍼      │
          │  27배 차이가 난다         │
          └─────────────────────────┘
```

---

# 26. 반드시 구분할 것

```text
용량 부족  ≠  inode 부족
   df -h        df -i

du  ≠  df
   파일을 셈     블록을 셈
   다르면 lsof +L1

write() 성공  ≠  디스크에 저장됨
   페이지 캐시를 거친다. sync / fsync

파일 삭제  ≠  공간 반환
   열고 있는 프로세스가 있으면 반환 안 됨

rootfs  ≠  데이터 파티션
   섞어 두면 꽉 찬 순간 복구도 못 한다

microSD  ≠  NVMe
   쓰기 수명이 10배 이상 차이

전부 기록  ≠  안전한 기록
   687 GB/일. 카드가 몇 주 만에 죽는다

기록 중지  ≠  데이터 보존
   sync 없이 전원을 끊으면 마지막 몇 초는 없다
```

---

# 27. Chapter 연결

```text
Chapter 4
Jetson / JetPack — eMMC, NVMe, microSD 부팅 구성

Chapter 12
Power & Thermal — 전력도 예산, 저장장치도 예산

Chapter 13  ← 여기
Storage & Data Logging

Chapter 15
Real-Time — 디스크 I/O가 제어 루프를 밀어내는 이유

Chapter 18
Observability — 디스크 사용량을 "미리" 감시하기

Chapter 19
Reliability — 사고 순간만 남기는 트리거 기반 기록
```

저장 설계는 결국 하나의 질문으로 압축된다.
**"이 데이터를 나중에 실제로 열어볼 것인가?"**
열어보지 않을 데이터는 기록하지 않는 것이 가장 빠르고 안전하다.
