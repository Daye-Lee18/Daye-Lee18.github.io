---
title: "Chapter 6. ROS 2 저장소 안에서 Rust 쓰기"
importance: 7
---

> **Goal:** 이미 C++/ROS 2로 돌아가는 저장소에 Rust workspace를 얹을 때
> 폴더를 어떻게 두고, `colcon`과 `cargo`를 어떻게 공존시키고,
> C++에서 Rust 함수를 부르려면 무엇이 필요한지 안다.

Chapter 1~5는 Rust만 있는 세상이었다.
실제로는 이미 C++ 노드와 `colcon`이 돌아가는 저장소에 Rust를 끼워 넣게 된다.
이 chapter는 그 경계를 다룬다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo build --release` · `colcon build` · `nm -D`

| 명령어                                             | 하는 일                         |
| :------------------------------------------------- | :------------------------------ |
| `cargo build --release`                            | Rust 쪽 빌드                    |
| `colcon build --symlink-install`                   | ROS 2 쪽 빌드                   |
| `colcon build --packages-select <pkg>`             | 특정 ROS package만              |
| `file target/release/libfoo.so`                    | 아키텍처 확인                   |
| `nm -D --defined-only target/release/libfoo.so`    | **내보낸 심볼 목록**            |
| `ldd target/release/libfoo.so`                     | 링크된 라이브러리               |
| `cargo build --target aarch64-unknown-linux-gnu`   | Jetson용                        |
| `cbindgen --config cbindgen.toml -o include/foo.h` | Rust → C 헤더 생성              |
| `cargo tree -e no-dev`                             | 배포에 실제로 들어가는 의존성만 |

---

# 1. 두 빌드 시스템이 한 저장소에 있다

```text
SeoulDynamics.SLAM/
├── Cargo.toml              ← Rust workspace 루트
├── Cargo.lock
├── src/                    ← Rust crate 들
│   ├── interfaces/
│   ├── fusion/
│   └── estimation/
├── ros2/                   ← ROS 2 package 들 (C++)
│   ├── slam_ros2/
│   │   ├── package.xml
│   │   ├── CMakeLists.txt
│   │   └── src/slam_node.cpp
│   └── ...
├── config/
├── scripts/
└── target/                 ← cargo 산출물   (gitignore)
    build/  install/  log/  ← colcon 산출물  (gitignore)
```

둘은 서로 모른다. **각자 자기 것만 빌드한다.**

```text
cargo   Cargo.toml 을 읽고  src/  를 빌드  →  target/
colcon  package.xml 을 읽고 ros2/ 를 빌드  →  build/ install/
```

`.gitignore`에 양쪽을 다 넣어야 한다.

```gitignore
# Rust
/target
# colcon
/build
/install
/log
```

---

# 2. 왜 Rust 코드를 `src/` 밑에 몰아 두나

`colcon`은 저장소를 훑으면서 `package.xml`이 있는 폴더를 ROS package로 인식한다.
Rust crate에는 `package.xml`이 없으니 그냥 지나친다. 그래서 섞여 있어도 충돌은 안 난다.

그래도 나눠 두는 편이 낫다.

```text
· 사람이 보기에 경계가 분명하다
· colcon 의 탐색 범위를 좁힐 수 있다
· Rust workspace 의 members 경로가 단순해진다
```

Chapter 1 §6에서 본 `[lib] path = "lib.rs"`가 필요했던 이유가 여기 있다.
기능별 폴더 구조(`src/interfaces/`, `src/fusion/`)를 먼저 정해 두고
Rust를 나중에 얹었기 때문에 Cargo 관례(`src/lib.rs`)와 어긋난 것이다.

```text
Rust 관례대로 갔다면          지금 구조
src/interfaces/               src/interfaces/
└── src/lib.rs                └── lib.rs      ← [lib] path 필요
```

**둘 다 정상이다.** 저장소 전체의 일관성을 택한 결과다.

---

# 3. Rust 코드를 어떻게 쓸 것인가 — 세 가지 길

```text
① 독립 실행파일          Rust binary 를 그냥 실행. ROS 2 와 무관
                        오프라인 도구, bag 분석기 등

② C++ 에서 라이브러리로 호출   Rust 를 cdylib/staticlib 으로 빌드하고
                        기존 C++ 노드가 링크해서 쓴다
                        → 지금 구조에 가장 잘 맞는다

③ Rust 가 직접 ROS 2 노드   r2r, rclrs 같은 crate 사용
                        → 아직 생태계가 얇다. 신중하게
```

②가 현실적인 이유는 이렇다.

```text
ROS 2 인터페이스(토픽·파라미터·TF)는 이미 C++ 쪽에 다 있다
Rust 로 옮기고 싶은 것은 보통 "계산" 부분이다
    → 계산만 Rust 라이브러리로 만들고 C++ 이 부른다
```

---

# 4. ②를 위한 설정 — `crate-type`

Chapter 2 §5에서 본 것이 여기서 쓰인다.

```toml
# src/fusion/Cargo.toml
[package]
name = "slam_fusion"

[lib]
path = "lib.rs"
crate-type = ["staticlib"]     # 또는 ["cdylib"]
```

```text
rlib        Rust 끼리만. C++ 에서는 못 쓴다
staticlib   .a  → C++ 바이너리에 통째로 들어간다
cdylib      .so → 런타임에 로드된다
```

어느 쪽을 쓸지는 배포 방식으로 갈린다.

```text
staticlib   장점: 배포가 단순 (파일 하나). .so 를 안 챙겨도 됨
            단점: 바이너리가 커진다. Rust 를 고치면 C++ 도 다시 링크

cdylib      장점: Rust 만 바꿔서 교체 가능
            단점: 런타임에 .so 를 찾아야 한다 (LD_LIBRARY_PATH)
```

로봇에 올릴 때는 **`staticlib`이 사고가 적다.** 챙길 파일이 하나 줄어든다.

---

# 5. C ABI로 함수 내보내기

Rust 함수를 그냥 만들면 C++에서 못 부른다. 이름이 mangling되고 호출 규약도 다르다.

```rust
// src/fusion/lib.rs

#[repr(C)]                       // C 와 같은 메모리 배치로
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[unsafe(no_mangle)]             // 이름을 그대로 유지 (edition 2024)
pub extern "C" fn fuse_step(     // C 호출 규약
    imu_ax: f64,
    imu_ay: f64,
    dt: f64,
) -> Pose {
    Pose { x: imu_ax * dt, y: imu_ay * dt, yaw: 0.0 }
}
```

세 가지가 필요하다.

```text
#[repr(C)]              구조체를 C 와 같은 배치로. Rust 기본 배치는 보장되지 않는다
#[unsafe(no_mangle)]    심볼 이름을 fuse_step 그대로 남긴다
extern "C"              C 호출 규약을 쓴다
```

**edition에 따라 표기가 다르다.** 예전 글을 그대로 따라 하면 여기서 막힌다.

```rust
// edition 2015 / 2018 / 2021
#[no_mangle]

// edition 2024  ← cargo init / cargo new 의 현재 기본값
#[unsafe(no_mangle)]
```

2021로 만든 프로젝트에서 `#[unsafe(no_mangle)]`을 쓰면 안 되고,
2024에서 `#[no_mangle]`을 쓰면 이렇게 거부된다.

```text
error: unsafe attribute used without unsafe
  |
4 | #[no_mangle]
  |   ^^^^^^^^^ usage of unsafe attribute
  |
help: wrap the attribute in `unsafe(...)`
  |
4 | #[unsafe(no_mangle)]
```

심볼 이름을 바꾸는 일은 링커 수준에서 충돌을 일으킬 수 있어서
2024부터 `unsafe`로 명시하게 바뀌었다. **하는 일은 똑같다.**
지금 내 edition은 `Cargo.toml`에서 확인한다. (Chapter 2 §3)

빌드하고 심볼이 실제로 나왔는지 확인한다.

```bash
cargo build --release

# Linux
nm -D --defined-only target/release/libslam_fusion.so | grep fuse_step
# 0000000000003a20 T fuse_step

# macOS 에서 확인할 때
nm -gU target/release/libslam_fusion.dylib | grep fuse_step
# 0000000000003fb4 T _fuse_step      ← macOS 는 앞에 _ 가 붙는다
```

**`nm`으로 확인하는 습관이 중요하다.** `no_mangle`을 빼먹으면
심볼이 **내보내기 목록에서 아예 사라지거나** `_ZN11slam_fusion9fuse_step17h...`
같은 이름으로 바뀌어서, C++ 링커가 `undefined reference to fuse_step`을 낸다.

---

# 6. C++ 쪽에서 부르기

헤더를 손으로 쓰거나 `cbindgen`으로 생성한다.

```c
/* include/slam_fusion.h */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { double x, y, yaw; } Pose;
Pose fuse_step(double imu_ax, double imu_ay, double dt);

#ifdef __cplusplus
}
#endif
```

자동 생성이 안전하다.

```bash
cargo install cbindgen
cbindgen --lang c --output include/slam_fusion.h src/fusion
```

`CMakeLists.txt`에서 링크한다.

```cmake
add_executable(slam_node src/slam_node.cpp)

target_include_directories(slam_node PRIVATE
  ${CMAKE_SOURCE_DIR}/../../include)

target_link_libraries(slam_node
  ${CMAKE_SOURCE_DIR}/../../target/release/libslam_fusion.a
  pthread dl m)          # Rust 런타임이 요구하는 것들
```

`pthread dl m`을 빼먹으면 링크 에러가 난다. Rust 표준 라이브러리가 쓰는 심볼들이다.

---

# 7. 빌드 순서 — Rust가 먼저다

`colcon`은 Rust를 모르므로 순서를 사람이 잡아줘야 한다.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ① Rust 먼저 — C++ 이 링크할 .a / .so 를 만든다
cargo build --release

# ② 그 다음 ROS 2
colcon build --symlink-install
source install/setup.bash
```

`scripts/build.sh` 같은 파일에 넣어두고 이것만 부르게 한다.
**순서를 잊어버려서 "예전 .a가 링크되는" 사고가 가장 흔하다.**

CMake에서 아예 `cargo`를 부르게 할 수도 있다.

```cmake
add_custom_target(rust_lib ALL
  COMMAND cargo build --release
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../..)
add_dependencies(slam_node rust_lib)
```

편하지만 `colcon`의 병렬 빌드와 겹치면 `cargo`가 여러 번 동시에 뜰 수 있다.
처음에는 스크립트로 순서를 명시하는 쪽이 디버깅하기 쉽다.

---

# 8. Docker 안에서 — Edge Computing Chapter 9.5와 이어서

`/ws`가 bind mount라는 사실이 Rust에도 그대로 적용된다.

```text
이미지에 굽는다              마운트한다 (/ws)
─────────────────           ──────────────────────
Rust 툴체인 (rustup)         내 crate 소스
cargo 자체                   Cargo.toml / Cargo.lock
자주 쓰는 cargo 확장          target/       ← 호스트에 남는다
                            build/ install/
```

`target/`이 `/ws` 아래에 있으므로 **컨테이너를 껐다 켜도 다시 빌드하지 않는다.**
`install/`이 `/ws`에 있어야 하는 이유와 정확히 같다.

Dockerfile에는 툴체인만 넣는다.

```dockerfile
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.82.0 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV CARGO_HOME=/usr/local/cargo
```

Chapter 9.5 §2의 표에서 `/usr/local/cargo/`가 "이미지 안에만 있다"고 되어 있던 것이 이것이다.

한 가지 주의할 점이 있다.

```text
호스트(x86 macOS/Ubuntu)에서 cargo build 를 한 적이 있으면
target/ 에 그 아키텍처 산출물이 남아 있다
    → 컨테이너(ARM 또는 다른 glibc)에서 그대로 쓰면 깨진다
```

Chapter 9.5 §27의 `install/` 문제와 똑같은 사고다. **빌드는 한쪽에서만 한다.**

```bash
rm -rf target && cargo build --release   # 컨테이너 안에서
```

---

# 9. 아키텍처 확인

Edge Computing Chapter 2의 내용이 그대로 필요하다.

```bash
file target/release/libslam_fusion.so
# ELF 64-bit LSB shared object, x86-64        ← 호스트에서 빌드함
# ELF 64-bit LSB shared object, ARM aarch64   ← Jetson 용

uname -m        # 지금 machine
```

Jetson용을 만드는 두 가지 길:

```text
① 크로스 컴파일   rustup target add aarch64-unknown-linux-gnu
                 링커와 sysroot 를 따로 챙겨야 한다. 설정이 번거롭다

② Jetson 용 컨테이너 안에서 그냥 빌드
                 느리지만 확실하다. 실무에서 더 많이 쓴다
```

---

# 10. 어디서부터 Rust로 옮길까

한 번에 다 바꾸는 것은 위험하다. 경계가 명확한 것부터 고른다.

```text
옮기기 좋은 것
  · 순수 계산 (필터, 기하 변환, 좌표 변환)
  · 입출력이 값 타입으로 딱 떨어지는 것
  · 테스트를 쓰기 쉬운 것
  · 이미 C++ 에서 자주 깨지던 것 (인덱스, 수명, 스레드)

나중으로 미룰 것
  · ROS 2 인터페이스 자체 (토픽·파라미터·TF)
  · 드라이버, 하드웨어 접근
  · 서드파티 C++ 라이브러리에 깊게 묶인 것
```

`interfaces` crate처럼 **공통 타입부터 정의하고 계산 모듈을 하나씩 붙이는** 순서가
Chapter 1 §8의 의존 방향과도 맞는다.

```text
       interfaces          가장 먼저. 아무것도 의존하지 않는다
        ↑     ↑
   fusion   estimation     계산 모듈. 하나씩 옮긴다
        ↑     ↑
      C++ ROS 2 노드        마지막까지 C++ 이어도 된다
```

---

# 11. Mini Practice

```bash
# 1) 간단한 staticlib 을 만들어 본다
mkdir -p /tmp/ffi_demo && cd /tmp/ffi_demo
cargo init --lib --name ffi_demo
cat >> Cargo.toml <<'TOML'

[lib]
crate-type = ["staticlib", "cdylib", "rlib"]
TOML
grep edition Cargo.toml        # cargo init 은 요즘 2024 를 쓴다

cat > src/lib.rs <<'RUST'
#[repr(C)]
pub struct Pose { pub x: f64, pub y: f64 }

#[unsafe(no_mangle)]          // edition 2021 이면 #[no_mangle]
pub extern "C" fn make_pose(x: f64, y: f64) -> Pose { Pose { x, y } }
RUST
cargo build --release

# 2) 무엇이 만들어졌나
ls target/release/ | grep -E '\.(a|so|dylib)$'

# 3) 심볼이 제대로 나왔나  (Linux: nm -D / macOS: nm -gU)
nm -gU target/release/libffi_demo.dylib 2>/dev/null | grep -i pose \
  || nm -D --defined-only target/release/libffi_demo.so | grep -i pose
#    T _make_pose   (macOS)  /  T make_pose  (Linux)

# 4) no_mangle 을 지우고 다시 — 심볼이 사라지는 것을 확인
sed -i.bak 's/#\[unsafe(no_mangle)\]//' src/lib.rs
cargo build --release
nm -gU target/release/libffi_demo.dylib 2>/dev/null | grep -i pose \
  || nm -D --defined-only target/release/libffi_demo.so | grep -i pose
#    아무것도 안 나온다  ← C++ 링커가 못 찾는 이유

# 5) 일부러 틀려 보기 — edition 2024 에서 #[no_mangle]
sed -i.bak2 's/#\[unsafe(no_mangle)\]/#[no_mangle]/' src/lib.rs 2>/dev/null
cargo build --release 2>&1 | head -8
#    error: unsafe attribute used without unsafe

# 6) 아키텍처 확인
file target/release/libffi_demo.*
```

4번에서 **`make_pose`가 목록에서 통째로 사라지는 것**을 보는 게 목적이다.
C++ 쪽에서는 이게 `undefined reference to 'make_pose'`로 나타난다.

---

# 12. 오늘의 핵심

```text
   SeoulDynamics.SLAM/

   Cargo.toml ──▶ cargo build --release ──▶ target/release/libX.a
        │                                        │
     src/*/                                      │  ① 먼저
                                                 ▼
   ros2/*/package.xml ──▶ colcon build ──▶ install/
        │                                   ② 나중에 링크
   CMakeLists.txt

   경계에 필요한 것
     #[repr(C)]      메모리 배치를 맞춘다
     #[no_mangle]    심볼 이름을 유지한다
     extern "C"      호출 규약을 맞춘다
     crate-type      staticlib / cdylib
     nm 으로 확인     심볼이 진짜 나왔는지
```

---

# 13. 반드시 구분할 것

```text
cargo  ≠  colcon
   서로 모른다. 순서를 사람이 잡아준다 (Rust 먼저)

rlib  ≠  staticlib / cdylib
   rlib 은 Rust 전용. C++ 에서 못 쓴다

no_mangle 없음  →  C++ 링커가 못 찾는다
   nm 으로 확인하는 습관

#[no_mangle]  vs  #[unsafe(no_mangle)]
   edition 2021 이하 vs edition 2024. 하는 일은 같다
   cargo init 의 기본값이 2024 라서 예전 글을 따라 하면 막힌다

#[repr(C)] 없음  →  필드 순서가 보장되지 않는다
   컴파일은 되는데 값이 이상하게 나온다. 더 무섭다

호스트에서 빌드한 target/  ≠  컨테이너에서 쓸 수 있는 target/
   Edge Computing Chapter 9.5 의 install/ 문제와 동일

staticlib  vs  cdylib
   배포 단순함 vs 교체 편함. 로봇은 보통 staticlib

Rust 관례(src/lib.rs)  vs  저장소 일관성(lib.rs)
   후자를 택하면 [lib] path 를 적으면 된다. 둘 다 정상
```

---

# 14. Chapter 연결

```text
Chapter 1
crate 의존 방향 — interfaces 를 맨 아래에 두는 이유

Chapter 2
crate-type — C++ 에서 부르려면 rlib 으로는 안 된다

Chapter 3
--release / 크로스 컴파일

Chapter 6  ← 여기
두 빌드 시스템의 공존과 FFI 경계

Edge Computing Chapter 2
ARM vs x86 — file 로 아키텍처 확인

Edge Computing Chapter 9.5
/ws 마운트 — target/ 이 호스트에 남는 이유

(다음)
소유권 · 빌림 · 라이프타임 — 이제 언어 자체로 들어간다
```
