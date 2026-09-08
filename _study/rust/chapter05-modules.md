---
title: "Chapter 5. Module 시스템 — crate 안으로"
importance: 6
---

> **Goal:** `mod` / `pub` / `use`가 각각 무엇을 하는지 구분하고,
> 파일과 module이 어떻게 대응되는지, `crate::` / `super::` / `self::`를 언제 쓰는지 안다.

여기서 처음으로 crate 안으로 들어간다.
Chapter 1에서 "module은 컴파일 단위가 아니라 이름공간"이라고 했다.
그러면 파일을 나누는 것과 module을 나누는 것은 무슨 관계일까?

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo check` · `cargo doc --open` · `cargo modules`

| 명령어                       | 하는 일                                             |
| :--------------------------- | :-------------------------------------------------- |
| `cargo check`                | module 경로가 맞는지 가장 빨리 확인                 |
| `cargo doc --open`           | **공개된(pub) 구조를 눈으로 본다**                  |
| `cargo doc --no-deps --open` | 내 crate 문서만                                     |
| `cargo tree`                 | crate 사이 관계 (module 아님)                       |
| `cargo expand`               | 매크로가 펼쳐진 결과 (`cargo install cargo-expand`) |
| `cargo modules structure`    | module 트리 (`cargo install cargo-modules`)         |
| `cargo clippy`               | `pub` 남용 등도 지적해준다                          |

---

# 1. 파일을 나눠도 crate는 하나다

먼저 Chapter 1 §7을 다시 확인한다.

```text
slam_interfaces crate

  lib.rs
   ├── mod sensor;      → sensor.rs
   ├── mod robot;       → robot.rs
   └── mod estimate;    → estimate.rs

        파일 4개 → rustc 호출 1번 → 산출물 1개
```

C++와 다른 지점이다.

```text
C++     헤더를 #include 한다. 파일마다 따로 컴파일된다
Rust    mod 로 "이 파일도 내 crate 의 일부"라고 선언한다
        include 가 아니라 트리에 붙이는 것에 가깝다
```

**Rust에는 `#include`가 없다.** 파일은 `mod` 선언으로만 crate 트리에 들어온다.
파일을 만들어 놓고 `mod`를 안 쓰면 **그 파일은 컴파일조차 되지 않는다.**
오타가 있어도 에러가 안 난다. 처음에 이걸로 한참 헤맨다.

---

# 2. `mod` — 트리에 붙이기

```text
src/
├── lib.rs
├── sensor.rs
└── robot.rs
```

```rust
// lib.rs
mod sensor;      // "sensor.rs 를 sensor module 로 붙여라"
mod robot;
```

`mod sensor;` 한 줄의 뜻은 이것이다.

```text
"sensor.rs (또는 sensor/mod.rs) 를 찾아서
 내 하위 module 로 트리에 붙여라"
```

인라인으로 쓸 수도 있다. 짧으면 이쪽이 낫다.

```rust
// lib.rs
mod sensor {
    pub struct Imu { pub stamp: f64 }
}
```

파일로 나누든 인라인으로 쓰든 **module 트리는 똑같다.**
파일 분리는 순전히 사람이 읽기 편하자고 하는 것이다.

---

# 3. 폴더로 나눌 때 — 두 가지 방식

module 안에 또 module을 두려면 폴더가 된다. 방식이 두 가지다.

```text
방식 A (요즘 권장)              방식 B (예전 방식)
src/                            src/
├── lib.rs                      ├── lib.rs
├── sensor.rs      ← 여기        └── sensor/
└── sensor/                         ├── mod.rs   ← 여기
    ├── imu.rs                      ├── imu.rs
    └── lidar.rs                    └── lidar.rs
```

```rust
// 방식 A: src/sensor.rs
pub mod imu;
pub mod lidar;

// 방식 B: src/sensor/mod.rs
pub mod imu;
pub mod lidar;
```

둘 다 동작한다. **방식 A가 권장**되는 이유는 `mod.rs` 파일이 여러 개 열려 있으면
에디터 탭에 전부 `mod.rs`로 보여서 구분이 안 되기 때문이다.

섞어 쓰면 에러가 난다.

```text
error: file for module `sensor` found at both "sensor.rs" and "sensor/mod.rs"
```

---

# 4. `pub` — 기본은 비공개

Rust는 **아무것도 안 붙이면 전부 private**다.

```rust
mod sensor {
    struct Imu;          // 이 module 안에서만
    pub struct Lidar;    // 밖에서도 보임
}
```

가시성은 계단식이다. `pub`을 하나만 빼먹어도 안 보인다.

```rust
mod sensor {             // ← private! 밖에서 sensor 자체가 안 보인다
    pub struct Lidar;    // 안이 pub 이어도 소용없다
}
```

```rust
pub mod sensor {         // ← 이렇게 해야 한다
    pub struct Lidar;
}
```

구조체 필드도 따로 `pub`이 필요하다.

```rust
pub struct Imu {
    pub stamp: f64,      // 밖에서 읽고 쓸 수 있다
    raw: [u8; 32],       // 밖에서 접근 불가
}
```

세밀하게 조절할 수도 있다.

```text
pub                 전부 공개
pub(crate)          이 crate 안에서만          ← 실무에서 가장 많이 쓴다
pub(super)          부모 module 까지만
pub(in path)        지정한 경로 안에서만
(없음)              이 module 안에서만
```

`pub(crate)`가 유용한 이유는 이렇다.

```text
crate 안의 다른 module 은 써야 하는데
다른 crate 에는 노출하고 싶지 않을 때
```

**`pub`을 남발하면 나중에 고칠 수 없게 된다.**
공개한 것은 남이 쓰고 있을 수 있으므로 함부로 못 바꾼다.
일단 `pub(crate)`로 두고 정말 필요할 때만 `pub`으로 올리는 편이 낫다.

---

# 5. `use` — 긴 경로를 줄이기

```rust
// use 없이
let imu = crate::sensor::imu::Imu::new();

// use 로 줄이면
use crate::sensor::imu::Imu;
let imu = Imu::new();
```

`use`는 **가져오는 게 아니라 별명을 만드는 것**에 가깝다.
`mod`가 트리에 붙이는 일이고, `use`는 그 트리의 한 지점에 짧은 이름을 다는 일이다.

```text
mod   →  트리를 만든다   (없으면 컴파일조차 안 된다)
use   →  이름을 줄인다   (없어도 긴 경로로 쓸 수 있다)
```

이 둘을 헷갈려서 `use`만 쓰고 `mod`를 빼먹는 실수가 아주 흔하다.

```rust
// lib.rs
use crate::sensor::Imu;    // ✗ 이것만으로는 sensor.rs 가 컴파일되지 않는다

mod sensor;                // ✓ 이게 있어야 한다
use crate::sensor::Imu;
```

여러 개를 묶어서 쓸 수 있다.

```rust
use crate::sensor::{imu::Imu, lidar::Lidar};
use std::collections::{HashMap, HashSet};

use crate::sensor::imu::Imu as ImuData;   // 이름 충돌 시 별명
```

---

# 6. 경로 — `crate::` / `super::` / `self::`

```text
crate::sensor::imu::Imu     crate 루트(lib.rs)부터 — 절대 경로
super::config::Config       부모 module 에서    — 상대 경로
self::helper::normalize     현재 module 에서    — 상대 경로
sensor::imu::Imu            std 나 외부 crate 이름으로 시작
```

예시 구조로 보면 이렇다.

```text
lib.rs (crate 루트)
├── mod sensor
│   ├── mod imu       ← 지금 여기서 코드를 쓴다면
│   └── mod lidar
└── mod fusion
```

```rust
// sensor/imu.rs 안에서
use crate::fusion::Filter;      // 루트부터 내려감
use super::lidar::Lidar;        // 형제 module (부모를 거쳐서)
use self::calib::apply;         // 내 하위 module

use std::collections::HashMap;  // 표준 라이브러리
use nalgebra::Vector3;          // 외부 crate (Cargo.toml 의 이름)
```

**팀 코드에서는 `crate::`로 시작하는 절대 경로를 권장**하는 경우가 많다.
파일을 옮겼을 때 `super::super::`가 깨지는 것보다 낫기 때문이다.

---

# 7. `pub use` — 다시 내보내기 (re-export)

내부 구조는 깊은데 쓰는 쪽에는 짧게 보여주고 싶을 때 쓴다.

```rust
// lib.rs
mod sensor;
mod fusion;

// 내부 경로를 crate 루트에서 바로 쓸 수 있게 올린다
pub use sensor::imu::Imu;
pub use sensor::lidar::Lidar;
pub use fusion::Filter;
```

```rust
// 쓰는 쪽
use slam_interfaces::Imu;                      // 짧다
// use slam_interfaces::sensor::imu::Imu;      // 이렇게 안 써도 된다
```

이게 라이브러리 설계에서 중요한 이유가 있다.

```text
공개 API 를 re-export 로 고정해두면
내부 폴더 구조를 나중에 바꿔도 쓰는 쪽 코드가 안 깨진다
```

`Measurement`, `Estimate`, `Stage` 같은 공통 타입을 담는 `interfaces` crate라면
`lib.rs`에서 `pub use`로 평평하게 내보내는 구성이 잘 맞는다.

---

# 8. `mod tests` — 테스트도 module이다

Chapter 3 §6에서 본 그 패턴이 사실은 module이다.

```rust
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;        // 부모 module 의 것을 전부 가져온다

    #[test]
    fn add_works() {
        assert_eq!(add(2, 3), 5);
    }
}
```

```text
#[cfg(test)]     테스트 빌드일 때만 컴파일하라
mod tests        평범한 하위 module 이다
use super::*     부모(= 지금 파일)의 항목을 전부 쓰겠다
```

`use super::*;`가 필요한 이유는 `tests`가 하위 module이라서
부모의 `add`가 자동으로 보이지 않기 때문이다.

private 함수도 테스트할 수 있다는 것이 이 방식의 장점이다.
같은 crate 안이므로 가시성 규칙에 걸리지 않는다.

---

# 9. 전형적인 crate 구조

```text
src/interfaces/
├── Cargo.toml
└── lib.rs
    │
    ├── pub mod sensor;      sensor.rs
    │     ├── pub mod imu;   sensor/imu.rs
    │     └── pub mod lidar; sensor/lidar.rs
    ├── pub mod estimate;    estimate.rs
    └── pub use ...          공개 API 를 평평하게
```

```rust
// lib.rs
pub mod sensor;
pub mod estimate;

pub use sensor::imu::Imu;
pub use sensor::lidar::Lidar;
pub use estimate::{Estimate, Stage};

// crate 안에서만 쓰는 것
pub(crate) mod util;
```

```rust
// 다른 crate 에서
use slam_interfaces::{Imu, Estimate, Stage};
```

---

# 10. Mini Practice

```bash
cd /tmp/rust_ws

# 1) module 을 나눠 본다
mkdir -p src/interfaces/sensor
cat > src/interfaces/lib.rs <<'RUST'
pub mod sensor;

pub use sensor::imu::Imu;

pub struct Measurement { pub stamp: f64 }
RUST

cat > src/interfaces/sensor.rs <<'RUST'
pub mod imu;
RUST

cat > src/interfaces/sensor/imu.rs <<'RUST'
pub struct Imu {
    pub stamp: f64,
    raw: [u8; 4],          // private
}

impl Imu {
    pub fn new(stamp: f64) -> Self {
        Self { stamp, raw: [0; 4] }
    }
}
RUST

cargo check -p demo_interfaces

# 2) re-export 덕분에 짧게 쓸 수 있는지 확인
cat > src/fusion/lib.rs <<'RUST'
use demo_interfaces::Imu;          // 짧은 경로 (pub use 덕분)
pub fn stamp_of(i: &Imu) -> f64 { i.stamp }
RUST
cargo check

# 3) private 필드는 못 건드린다 — 일부러 에러를 내 본다
cat >> src/fusion/lib.rs <<'RUST'
pub fn peek(i: &Imu) -> [u8; 4] { i.raw }   // ← 에러가 나야 정상
RUST
cargo check 2>&1 | head -8

# 4) mod 를 빼먹으면 어떻게 되나 — 두 가지 경우
#    (4-a) mod 만 지우고 pub use 는 남기면 → 에러
sed -i 's/^pub mod sensor;//' src/interfaces/lib.rs
cargo check -p demo_interfaces 2>&1 | grep '^error' | head -2
#    error[E0433]: failed to resolve: use of unresolved module ... `sensor`

#    (4-b) 둘 다 지우면 → 아무 말도 안 한다
cat > src/interfaces/lib.rs <<'RUST'
pub struct Measurement { pub stamp: f64 }
RUST
echo 'this is not valid rust at all !!!' >> src/interfaces/sensor/imu.rs
cargo check -p demo_interfaces
#    Finished — 문법이 완전히 깨진 파일이 있는데도 통과한다

# 5) 문서로 공개 구조 보기
cargo doc -p demo_interfaces --no-deps --open
```

**4-b가 이 chapter에서 가장 중요한 실험이다.**
`imu.rs`에 말도 안 되는 글자를 넣었는데 빌드가 성공한다.
`mod` 선언이 없으니 Rust는 그 파일의 존재 자체를 모르기 때문이다. (§1)

```text
파일을 만들었는데 코드가 안 도는 것 같다
  → 십중팔구 mod 선언을 빼먹은 것이다
  → 에러도 안 나므로 스스로 의심하는 수밖에 없다
```

3번의 `error[E0616]: field \`raw\` of struct \`Imu\` is private`도 같이 확인한다.

---

# 11. 오늘의 핵심

```text
                   crate = 컴파일 단위 하나

  ┌──────────────────────────────────────────────────┐
  │ lib.rs  (crate 루트)                              │
  │                                                   │
  │   mod sensor;   ──▶ 트리에 붙인다                  │
  │        │              (없으면 파일이 컴파일 안 됨)  │
  │        ├── mod imu                                │
  │        └── mod lidar                              │
  │                                                   │
  │   pub          ──▶ 밖에서 보이게 (기본은 private)   │
  │   pub(crate)   ──▶ 이 crate 안에서만               │
  │                                                   │
  │   use          ──▶ 이름을 줄인다 (선택사항)         │
  │   pub use      ──▶ 다시 내보낸다 (공개 API 설계)    │
  └──────────────────────────────────────────────────┘

        파일을 나눠도 rustc 호출은 여전히 1번
```

---

# 12. 반드시 구분할 것

```text
mod  ≠  use
   mod 는 트리에 붙인다 (필수)
   use 는 이름을 줄인다 (편의)
   mod 없이 use 만 쓰면 파일이 컴파일조차 안 된다

파일을 나눔  ≠  crate 를 나눔
   파일: 컴파일 단위 그대로 1개
   crate: 컴파일 단위가 늘어난다 (Chapter 1 §7)

#include  ≠  mod
   Rust 에는 include 가 없다. 트리에 붙이는 개념

pub 이 없으면 private
   기본이 비공개다. C++ 의 struct 와 반대

pub mod  ≠  pub struct
   module 이 private 이면 안의 pub 은 소용없다

pub  vs  pub(crate)
   일단 pub(crate) 로 두는 편이 안전하다

sensor.rs  vs  sensor/mod.rs
   둘 다 되지만 섞으면 에러. 요즘은 sensor.rs 권장

crate::  vs  super::
   절대 경로가 파일 이동에 강하다
```

---

# 13. Chapter 연결

```text
Chapter 1
crate 는 컴파일 단위, module 은 이름공간

Chapter 3
cargo doc — 공개 구조를 눈으로 확인

Chapter 5  ← 여기
mod / pub / use — crate 안을 정리하는 법

Chapter 6
ROS 2 저장소 안에서 Rust 쓰기

(그 다음)
소유권 · 빌림 · 라이프타임 — 드디어 언어 자체로
```
