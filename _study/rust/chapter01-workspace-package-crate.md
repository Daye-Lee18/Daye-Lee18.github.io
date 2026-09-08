---
title: "Chapter 1. Workspace, Package, Crate, Module"
importance: 2
---

> **Goal:** Rust 프로젝트의 네 단위를 구분하고, `Cargo.toml`이 왜 여러 개인지,
> `[workspace].members`에 등록하는 것이 정확히 무엇인지 설명할 수 있다.

Rust를 처음 만지면 문법보다 이게 먼저 막힌다.
`Cargo.toml`이 두 개고, crate라는 말이 package를 뜻하는 것 같기도 하고 파일을 뜻하는 것 같기도 하다.
이 chapter는 그 지도를 먼저 그린다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo check` · `cargo metadata` · `cargo tree`

| 명령어                                        | 하는 일                                              |
| :-------------------------------------------- | :--------------------------------------------------- |
| `cargo new my_pkg`                            | binary package 새로 만들기 (`main.rs`)               |
| `cargo new my_pkg --lib`                      | library package 새로 만들기 (`lib.rs`)               |
| `cargo check`                                 | 컴파일만 해보고 실행파일은 안 만듦. **가장 자주 침** |
| `cargo build`                                 | 실제로 빌드                                          |
| `cargo metadata --format-version 1 \| head`   | Cargo가 인식한 workspace 구조 전체                   |
| `cargo metadata --no-deps --format-version 1` | 내 package들만 (의존성 제외)                         |
| `cargo tree`                                  | 의존 관계 트리                                       |
| `cargo tree -p slam_interfaces`               | 특정 package 기준으로                                |
| `cargo build -p slam_interfaces`              | workspace 안에서 **그 package만** 빌드               |
| `cargo build --workspace`                     | 전체 member 빌드                                     |
| `cargo locate-project`                        | 지금 위치가 어느 `Cargo.toml`에 속하는지             |

---

# 1. 네 단어, 네 층

먼저 결론부터 본다. 이 네 개는 서로 포함 관계다.

```text
Workspace     여러 package를 한 번에 관리하는 묶음
   └── Package        Cargo.toml 하나가 정의하는 단위
          └── Crate          rustc가 한 번에 컴파일하는 단위
                 └── Module        crate 안에서 코드를 나누는 이름공간
```

각각을 한 문장으로 정리하면 이렇다.

```text
Workspace   "이 폴더들을 같이 빌드하고, 의존성 버전도 같이 맞추자"
Package     "이건 배포·버전 관리의 단위다"           → Cargo.toml 이 있으면 package
Crate       "이건 컴파일의 단위다"                  → lib.rs 또는 main.rs 가 시작점
Module      "이건 그냥 코드 정리다"                 → 컴파일 단위가 아니다
```

가장 중요한 구분은 **Package와 Crate**다. 둘을 같은 말로 쓰는 글이 많아서 헷갈리는데,
**하나의 package가 여러 crate를 가질 수 있다.** (§5)

---

# 2. ROS 2를 안다면 이렇게 대응된다

C++/ROS 2를 먼저 했다면 아래 대응이 가장 빠르다.

| Rust                   | ROS 2 / C++                                 | 공통점                           |
| :--------------------- | :------------------------------------------ | :------------------------------- |
| Workspace              | colcon workspace                            | 여러 package를 한 번에 빌드      |
| Package (`Cargo.toml`) | ROS 2 package (`package.xml`)               | 이름·버전·의존성을 선언하는 단위 |
| Crate                  | `.so` / `.a` 라이브러리 하나, 실행파일 하나 | 링크되는 산출물 하나             |
| Module (`mod`)         | C++ `namespace`                             | 이름 충돌만 막는 논리적 분리     |

완전히 같지는 않지만 첫 감을 잡는 데는 충분하다.
특히 **module은 `namespace`처럼 파일을 나눠도 컴파일 산출물은 하나**라는 점이 같다.

---

# 3. `Cargo.toml`이 두 개인 이유

실제 구조를 보자.

```text
SeoulDynamics.SLAM/
├── Cargo.toml              ← ① workspace 관리용
└── src/
    └── interfaces/
        ├── Cargo.toml      ← ② 실제 package 설정
        └── lib.rs          ← ③ library crate 의 시작점
```

두 파일은 역할이 완전히 다르다.

**① 루트 `Cargo.toml`**

```toml
[workspace]
members = ["src/interfaces"]
resolver = "2"
```

여기에는 `[package]`가 **없다.**
그래서 **루트 자체는 package도 아니고 crate도 아니다.** 빌드되는 코드가 없다.
하는 일은 이것뿐이다.

```text
· 어떤 하위 package 들을 묶을지 지정
· 공통 버전 / edition / license 를 한곳에 정의
· target/ 과 Cargo.lock 을 공유
· cargo build 한 번으로 전부 빌드
```

**② `src/interfaces/Cargo.toml`**

```toml
[package]
name = "slam_interfaces"
version.workspace = true
edition.workspace = true

[lib]
path = "lib.rs"
```

이쪽에 `[package]`가 있다. **이게 진짜 package다.**
package 이름은 폴더 이름(`interfaces`)이 아니라 여기 적은 `slam_interfaces`다.

---

# 4. `members`에 등록하는 것은 crate가 아니라 package 경로다

여기가 가장 자주 헷갈리는 지점이다.

```toml
[workspace]
members = [
    "src/interfaces",     # ← 폴더 경로다
    "src/fusion",
]
```

crate 이름을 적는 게 아니다. **`Cargo.toml`이 들어 있는 폴더의 경로**를 적는다.

```text
[workspace].members
      │  폴더 경로를 등록
      ▼
   그 폴더의 Cargo.toml  →  package 를 정의
      │
      ▼
   그 package 가  →  library crate 또는 binary crate 를 만든다
```

그래서 "crate를 members에 넣는다"는 표현은 부정확하다.
**등록하는 것은 package 경로, 만들어지는 것은 crate**다.

폴더를 만들고 `Cargo.toml`을 넣었는데 `members`에 안 적으면 어떻게 될까.
직접 해보면 위치에 따라 반응이 다르다.

```text
루트에서 cargo check       → 아무 말 없이 그 package 를 건너뛴다
                              (빌드 대상이 아니라고 판단)

그 폴더 안에서 cargo check → 에러
```

```text
error: current package believes it's in a workspace when it's not:
current:   /path/src/fusion/Cargo.toml
workspace: /path/Cargo.toml

this may be fixable by adding `src/fusion` to the `workspace.members`
array of the manifest located at: /path/Cargo.toml
```

Rust를 시작하고 며칠 안에 반드시 한 번은 보는 메시지다.
**"workspace 루트는 있는데 나는 그 명단에 없다"**는 뜻이고,
해결은 메시지가 알려주는 대로 `members`에 경로를 추가하는 것이다.

루트에서는 조용히 넘어가기 때문에 **빌드는 되는데 내 코드만 안 들어가는** 상황이 생긴다.
`cargo metadata`로 실제 인식된 목록을 확인하는 습관이 그래서 필요하다. (§9)

---

# 5. 하나의 package가 여러 crate를 가질 수 있다

Package와 crate가 1:1이 아니라는 것이 두 번째 핵심이다.

```text
하나의 package 는

  library crate   최대 1개    ← lib.rs
  binary crate    여러 개     ← main.rs, src/bin/*.rs
```

같이 두면 이렇게 된다.

```text
my_package/
├── Cargo.toml
├── lib.rs          → library crate  (다른 코드가 가져다 씀)
└── main.rs         → binary crate   (직접 실행함)
```

```toml
[package]
name = "my_package"

[lib]
path = "lib.rs"

[[bin]]
name = "my_app"
path = "main.rs"
```

`[lib]`는 하나뿐이라 대괄호가 하나(`[lib]`)이고,
`[[bin]]`은 여러 개일 수 있어서 대괄호가 두 개(`[[bin]]`)다. TOML의 배열 표기다.

```bash
cargo build              # 둘 다 빌드
cargo run --bin my_app   # binary 실행
```

라이브러리와 그 라이브러리를 쓰는 CLI 도구를 한 package에 두는 것이 흔한 구성이다.

---

# 6. 기본 경로와 직접 지정하는 경로

Cargo는 아무 설정이 없으면 정해진 위치를 찾는다. 이것이 관례(convention)다.

```text
관례를 따르는 구조                    Cargo 가 자동으로 인식
my_package/
├── Cargo.toml
└── src/
    ├── lib.rs        → library crate
    ├── main.rs       → binary crate (package 이름으로)
    └── bin/
        ├── tool_a.rs → binary crate "tool_a"
        └── tool_b.rs → binary crate "tool_b"
```

관례를 따르면 `Cargo.toml`에 `[lib]`나 `[[bin]]`을 쓸 필요조차 없다.

그런데 기존 저장소에 기능별 폴더 구조가 이미 있으면 관례와 안 맞을 수 있다.

```text
src/interfaces/
├── Cargo.toml
└── lib.rs          ← src/ 아래가 아니라 바로 여기 있다
```

이때 `path`로 직접 알려준다.

```toml
[lib]
path = "lib.rs"
```

**관례를 따르면 설정이 필요 없고, 벗어나면 설정으로 알려줘야 한다.**
`[lib] path = "lib.rs"`가 보이는 이유가 이것이다.

---

# 7. Crate가 "컴파일 단위"라는 말의 의미

crate 하나는 `rustc`를 **한 번** 호출해서 만들어진다.
파일이 몇 개든, module이 몇 겹이든 상관없다.

```text
slam_interfaces crate

  lib.rs
   ├── mod sensor;      sensor.rs
   ├── mod robot;       robot.rs
   └── mod estimate;    estimate.rs

        파일 4개
             ↓
        rustc 호출 1번
             ↓
        libslam_interfaces.rlib   ← 산출물 1개
```

C/C++와 다른 지점이 여기다.

```text
C++     .cpp 파일마다 컴파일 → .o 여러 개 → 링커가 묶음
Rust    crate 전체를 한 번에 컴파일 → 산출물 하나
```

그래서 Rust는 파일을 아무리 나눠도 컴파일 횟수가 늘지 않는다.
대신 **crate를 나누면 늘어난다.** 이것이 crate를 나누는 실질적인 이유 중 하나다.

```text
crate 를 나누면
  ✓ 바뀌지 않은 crate 는 다시 컴파일 안 한다  → 증분 빌드가 빨라진다
  ✓ 여러 crate 를 병렬로 컴파일할 수 있다
  ✗ 대신 crate 경계를 넘는 최적화는 줄어든다
```

---

# 8. Crate끼리 의존시키기

workspace 안의 crate는 서로 가져다 쓸 수 있다.

```text
slam_cpp_rs
  ├─ depends on  slam_fusion
  ├─ depends on  slam_estimation
  └─ depends on  slam_interfaces
```

의존은 그 package의 `Cargo.toml`에 적는다.

```toml
# src/fusion/Cargo.toml
[package]
name = "slam_fusion"

[dependencies]
slam_interfaces = { path = "../interfaces" }
```

그러면 코드에서 이렇게 쓴다.

```rust
use slam_interfaces::Measurement;
```

`use` 뒤에 오는 것은 폴더 이름(`interfaces`)이 아니라 **crate 이름(`slam_interfaces`)**이다.
헷갈리면 `Cargo.toml`의 `name`을 보면 된다.

`interfaces`가 공통 타입(`Measurement`, `Estimate`, `Stage`)을 담고
나머지가 그것을 가져다 쓰는 구조라면, 의존 방향이 한쪽으로만 흘러야 한다.

```text
       interfaces          ← 아무것도 의존하지 않는다
        ↑     ↑
   fusion   estimation
        ↑     ↑
       slam_cpp_rs
```

Rust는 **crate 사이의 순환 의존을 허용하지 않는다.**
`fusion`이 `interfaces`를 쓰는데 `interfaces`도 `fusion`을 쓰면 컴파일이 거부된다.
공통 타입 crate를 맨 아래에 따로 두는 이유가 이것이다.

---

# 9. 지금 구조를 눈으로 확인하기

머리로 그리지 말고 Cargo에게 물어보면 된다.

```bash
# 내 package 들만 (의존성 제외)
cargo metadata --no-deps --format-version 1 | python3 -m json.tool | head -40

# 이름과 경로만 뽑기
cargo metadata --no-deps --format-version 1 \
  | python3 -c "import sys,json; [print(p['name'], '→', p['manifest_path']) for p in json.load(sys.stdin)['packages']]"

# 지금 내 위치가 어느 Cargo.toml 소속인지
cargo locate-project

# 의존 트리
cargo tree
```

`cargo metadata`가 보여주는 것이 **Cargo가 실제로 인식한 구조**다.
내가 생각한 구조와 다르면 대부분 `members` 등록을 빠뜨린 것이다.

---

# 10. Mini Practice

작은 workspace를 직접 만들어 본다.

```bash
mkdir -p /tmp/rust_ws && cd /tmp/rust_ws

# 루트: workspace 만
cat > Cargo.toml <<'TOML'
[workspace]
resolver = "2"
members = ["src/interfaces", "src/fusion"]

[workspace.package]
version = "0.1.0"
edition = "2021"
TOML

# package 1: library
mkdir -p src/interfaces
cat > src/interfaces/Cargo.toml <<'TOML'
[package]
name = "demo_interfaces"
version.workspace = true
edition.workspace = true

[lib]
path = "lib.rs"
TOML
echo 'pub struct Measurement { pub stamp: f64 }' > src/interfaces/lib.rs

# package 2: interfaces 를 쓰는 library
mkdir -p src/fusion
cat > src/fusion/Cargo.toml <<'TOML'
[package]
name = "demo_fusion"
version.workspace = true
edition.workspace = true

[lib]
path = "lib.rs"

[dependencies]
demo_interfaces = { path = "../interfaces" }
TOML
cat > src/fusion/lib.rs <<'RUST'
use demo_interfaces::Measurement;
pub fn latest(m: &Measurement) -> f64 { m.stamp }
RUST

cargo check
cargo tree
cargo metadata --no-deps --format-version 1 \
  | python3 -c "import sys,json; [print(p['name'],'→',p['manifest_path']) for p in json.load(sys.stdin)['packages']]"
```

여기까지 하면 이렇게 나온다.

```text
$ cargo check
    Checking demo_interfaces v0.1.0 (/tmp/rust_ws/src/interfaces)
    Checking demo_fusion v0.1.0 (/tmp/rust_ws/src/fusion)
    Finished `dev` profile

$ cargo tree
demo_fusion v0.1.0
└── demo_interfaces v0.1.0

demo_interfaces v0.1.0
```

이제 §4를 몸으로 확인해 본다. `members`에서 `"src/fusion"`을 지우고:

```bash
# 루트에서 — 아무 말 없이 넘어간다
cargo check

# 그런데 그 폴더 안에서는 에러가 난다
cd src/fusion && cargo check
```

```text
error: current package believes it's in a workspace when it's not
```

**루트에서는 조용하고 안에서는 에러**라는 비대칭이 §4의 핵심이다.

---

# 11. 오늘의 핵심

```text
        Cargo.toml 이 있으면 package
        lib.rs / main.rs 가 있으면 crate

   ┌──────────────────────────────────────────┐
   │ Workspace   루트 Cargo.toml [workspace]   │
   │             빌드되는 코드는 없다            │
   │  ┌────────────────────────────────────┐  │
   │  │ Package   src/interfaces/Cargo.toml │  │
   │  │           name = "slam_interfaces"  │  │
   │  │  ┌──────────────────────────────┐  │  │
   │  │  │ Crate    lib.rs              │  │  │
   │  │  │          rustc 호출 1번       │  │  │
   │  │  │  ┌────────────────────────┐  │  │  │
   │  │  │  │ Module   mod sensor;   │  │  │  │
   │  │  │  │          이름공간일 뿐   │  │  │  │
   │  │  │  └────────────────────────┘  │  │  │
   │  │  └──────────────────────────────┘  │  │
   │  └────────────────────────────────────┘  │
   └──────────────────────────────────────────┘
```

---

# 12. 반드시 구분할 것

```text
Package  ≠  Crate
   package 1개가 lib crate 1개 + bin crate 여러 개를 가질 수 있다

[workspace].members 에 등록하는 것
   = package 폴더 경로       (crate 이름이 아니다)

폴더 이름  ≠  crate 이름
   src/interfaces/ 인데 이름은 slam_interfaces 일 수 있다
   use 에 쓰는 것은 Cargo.toml 의 name

루트 Cargo.toml  ≠  package
   [package] 가 없으면 빌드되는 코드가 없다

Crate 를 나눔  ≠  파일을 나눔
   파일을 나눠도 컴파일 단위는 그대로 1개
   crate 를 나눠야 컴파일 단위가 늘어난다

[lib]  vs  [[bin]]
   대괄호 1개 = 최대 하나 / 대괄호 2개 = 여러 개 (TOML 배열)
```

---

# 13. Chapter 연결

```text
Chapter 1  ← 여기
workspace / package / crate / module — 전체 지도

Chapter 2
Cargo.toml 을 한 줄씩 읽기 — workspace 상속, [lib], [[bin]]

Chapter 3
cargo 명령어와 target/ — 빌드가 실제로 무엇을 만드나

Chapter 4
의존성과 Cargo.lock

Chapter 5
module 시스템 — crate 안으로 들어간다

Chapter 6
ROS 2 저장소 안에서 Rust 쓰기
```
