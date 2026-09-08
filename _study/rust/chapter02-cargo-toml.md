---
title: "Chapter 2. Cargo.toml 한 줄씩 읽기"
importance: 3
---

> **Goal:** `Cargo.toml`의 각 섹션이 무엇을 결정하는지 알고,
> workspace 상속(`version.workspace = true`)과 `[lib]` / `[[bin]]` / `[dependencies]`를
> 직접 쓸 수 있다.

Chapter 1에서 `Cargo.toml`이 두 종류라는 것을 봤다.
이 chapter는 그 안을 한 줄씩 연다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo add` · `cargo metadata` · `cargo check`

| 명령어                                        | 하는 일                           |
| :-------------------------------------------- | :-------------------------------- |
| `cargo add serde`                             | `[dependencies]`에 자동으로 추가  |
| `cargo add serde --features derive`           | feature까지 지정해서 추가         |
| `cargo add --dev criterion`                   | `[dev-dependencies]`에 추가       |
| `cargo remove serde`                          | 의존성 제거                       |
| `cargo metadata --no-deps --format-version 1` | Cargo가 해석한 최종 설정          |
| `cargo check`                                 | 설정을 고친 뒤 가장 먼저          |
| `cargo verify-project`                        | `Cargo.toml` 문법이 맞는지만 확인 |
| `cargo run --bin <name>`                      | 특정 binary 실행                  |
| `cargo build -p <package>`                    | 특정 package만                    |

---

# 1. TOML 문법 최소한만

`Cargo.toml`은 TOML 형식이다. 세 가지만 알면 읽을 수 있다.

```toml
# ① [이름]  = 테이블 하나
[package]
name = "slam_interfaces"

# ② [[이름]] = 테이블 배열. 같은 이름을 여러 번 쓸 수 있다
[[bin]]
name = "tool_a"

[[bin]]
name = "tool_b"

# ③ { } = 인라인 테이블. 한 줄로 줄여 쓴 것
serde = { version = "1.0", features = ["derive"] }
```

②가 `[lib]`와 `[[bin]]`의 차이를 만든다.
library crate는 package당 최대 하나라서 `[lib]`,
binary crate는 여러 개라서 `[[bin]]`이다.

---

# 2. 루트 `Cargo.toml` — workspace 쪽

```toml
[workspace]
resolver = "2"
members = [
    "src/interfaces",
    "src/fusion",
]
exclude = ["experiments/scratch"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Proprietary"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
nalgebra = "0.33"
```

한 줄씩 보면 이렇다.

```text
resolver = "2"        feature 해석 방식. edition 2021 이면 "2" 를 쓴다
members               workspace 에 포함할 package 폴더 경로 (Chapter 1 §4)
exclude               안에 있지만 workspace 에 넣지 않을 폴더
[workspace.package]   하위 package 들이 상속할 공통 메타데이터
[workspace.dependencies]  하위 package 들이 상속할 공통 의존성 버전
```

`[workspace.package]`와 `[workspace.dependencies]`는 **정의만 해둘 뿐**이고,
자동으로 적용되지 않는다. 각 package가 "그거 쓸게"라고 명시해야 한다. (§4)

---

# 3. 하위 `Cargo.toml` — package 쪽

```toml
[package]
name = "slam_interfaces"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
path = "lib.rs"

[dependencies]
serde.workspace = true
slam_common = { path = "../common" }
```

```text
name        crate 이름이 된다. use 에 쓰는 이름 (폴더 이름이 아니다)
version     이 package 의 버전
edition     Rust 판본. 2015 / 2018 / 2021 / 2024
license     배포용 메타데이터
[lib]       library crate 설정
[dependencies]  이 package 가 쓰는 crate 들
```

`edition`은 문법 판본이다. Rust는 하위 호환을 깨지 않으려고
"새 문법은 새 edition에서만" 방식으로 간다.
**edition이 다른 crate끼리도 서로 의존할 수 있다.**

---

# 4. `.workspace = true` — 상속 문법

이게 처음 보면 낯설다.

```toml
version.workspace = true
```

이 한 줄의 뜻은 **"이 값은 내가 안 쓰고 루트 `[workspace.package]`에서 가져오겠다"**이다.

```text
루트 Cargo.toml                     하위 Cargo.toml
────────────────────────           ─────────────────────────
[workspace.package]                 [package]
version = "0.1.0"      ────────▶    version.workspace = true
edition = "2021"       ────────▶    edition.workspace = true
```

package가 10개일 때 버전을 올리려면 원래는 10군데를 고쳐야 한다.
상속을 쓰면 루트 한 곳만 고치면 된다.

의존성도 같은 방식이다.

```toml
# 루트
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }

# 하위
[dependencies]
serde.workspace = true
```

**버전이 갈리는 사고를 막는 것**이 진짜 목적이다.
package마다 `serde = "1.0"`, `serde = "1.0.190"`을 따로 적으면
서로 다른 버전이 동시에 링크되는 상황이 생길 수 있다.

일부만 덮어쓸 수도 있다.

```toml
[dependencies]
serde = { workspace = true, features = ["rc"] }   # 버전은 상속, feature 는 추가
```

---

# 5. `[lib]` — library crate 설정

```toml
[lib]
name = "slam_interfaces"     # 생략하면 package name 을 씀
path = "lib.rs"              # 생략하면 src/lib.rs
crate-type = ["rlib"]        # 생략하면 rlib
```

`crate-type`은 무엇으로 만들지를 정한다.

```text
rlib        Rust 전용 정적 라이브러리. 기본값. Rust 끼리 쓸 때
cdylib      C ABI 동적 라이브러리 (.so / .dylib / .dll)
                → C++/Python 에서 부를 때
staticlib   C ABI 정적 라이브러리 (.a)
                → C++ 링커에 넘길 때
proc-macro  컴파일 타임에 코드를 생성하는 특수 crate
```

ROS 2의 C++ 노드에서 Rust 코드를 부르고 싶다면 `rlib`으로는 안 된다.
`cdylib`이나 `staticlib`이 필요하다. (Chapter 6)

```toml
[lib]
crate-type = ["rlib", "cdylib"]    # 둘 다 만들 수도 있다
```

---

# 6. `[[bin]]` — 실행파일

```toml
[[bin]]
name = "slam-node"
path = "main.rs"

[[bin]]
name = "bag-dump"
path = "tools/bag_dump.rs"
```

```bash
cargo run --bin slam-node
cargo run --bin bag-dump
cargo build --bins              # 모든 binary
```

관례를 따르면 `[[bin]]`을 아예 안 써도 된다.

```text
<package 루트>/src/main.rs          → package 이름과 같은 binary
<package 루트>/src/bin/foo.rs       → binary "foo"
<package 루트>/src/bin/bar/main.rs  → binary "bar"
```

**경로가 package 루트 기준**이라는 점이 중요하다. workspace 루트가 아니다.

여기서 한 번 걸린다. Chapter 1의 구조처럼 `lib.rs`를 package 루트에 바로 두면
(관례인 `src/lib.rs`가 아니라) `[lib] path`를 적어야 했다. 그런데 binary는 그것과 무관하게
**여전히 `src/bin/`을 찾는다.**

```text
src/fusion/                    ← package 루트
├── Cargo.toml
├── lib.rs                     ← 관례 밖. [lib] path = "lib.rs" 필요
├── bin/demo.rs                ✗ 여기는 안 본다
└── src/
    └── bin/demo.rs            ✓ 여기를 본다
```

`src/fusion/bin/demo.rs`에 두고 `cargo run --bin demo`를 하면 이렇게 나온다.

```text
error: no bin target named `demo` in `demo_fusion` package
```

한 package 안에서 관례를 반만 따르면 이런 어긋남이 생긴다.
그래서 이런 구조에서는 binary도 명시하는 편이 헷갈리지 않는다.

```toml
[[bin]]
name = "demo"
path = "tools/demo.rs"
```

**관례를 따를 거면 전부 따르고, 벗어날 거면 전부 명시한다.** 반씩 섞는 것이 제일 나쁘다.

---

# 7. `[dependencies]` — 세 가지 출처

```toml
[dependencies]
# ① crates.io 에서
serde = "1.0"

# ② 같은 저장소 안에서 (경로)
slam_interfaces = { path = "../interfaces" }

# ③ git 에서
some_crate = { git = "https://github.com/user/repo", tag = "v0.3.0" }
```

로컬 개발은 ②를 쓰고, 배포할 때는 ①로 바꾸는 것이 보통이다.
③은 편하지만 **재현성이 약하다.** 반드시 `tag`나 `rev`로 고정한다.

```toml
# 나쁨 — 브랜치가 움직이면 빌드 결과가 달라진다
some_crate = { git = "...", branch = "main" }

# 좋음 — 커밋이 고정된다
some_crate = { git = "...", rev = "a1b2c3d" }
```

---

# 8. 의존성의 세 종류

```toml
[dependencies]              # 빌드 결과물에 들어간다
nalgebra = "0.33"

[dev-dependencies]          # 테스트 · 예제 · 벤치마크에만
criterion = "0.5"

[build-dependencies]        # build.rs 에서만
cc = "1.0"
```

`dev-dependencies`는 **최종 산출물에 들어가지 않는다.**
테스트용 무거운 crate를 여기에 넣으면 배포 바이너리가 커지지 않는다.

`build-dependencies`는 `build.rs`(빌드 전에 돌아가는 Rust 스크립트)에서 쓴다.
C 라이브러리를 같이 컴파일하거나 코드 생성을 할 때 필요하다.

---

# 9. `[profile]` — 최적화 설정

```toml
[profile.dev]
opt-level = 0        # 최적화 없음. 컴파일이 빠름
debug = true

[profile.release]
opt-level = 3        # 최대 최적화
lto = true           # link-time optimization
codegen-units = 1    # 병렬 코드 생성을 포기하고 최적화를 얻음
strip = true         # 디버그 심볼 제거 → 파일 크기 감소
```

`profile`은 **workspace 루트에만** 쓸 수 있다.
하위 package에 적으면 무시되고 경고가 뜬다.

로봇에 올릴 바이너리라면 `release`가 사실상 필수다.
`dev` 빌드는 최적화가 없어서 수십 배 느릴 수 있다. (Chapter 3 §5)

디버깅이 필요한 배포판은 이렇게 절충한다.

```toml
[profile.release]
debug = true         # 심볼은 남기고 최적화는 그대로
strip = false
```

---

# 10. `[features]` — 선택적 기능

```toml
[features]
default = ["std"]
std = []
ros = ["dep:r2r"]
visualize = ["dep:plotters"]

[dependencies]
r2r = { version = "0.9", optional = true }
plotters = { version = "0.3", optional = true }
```

```bash
cargo build                          # default feature
cargo build --features ros           # ros 추가
cargo build --no-default-features    # default 끄기
cargo build --all-features           # 전부 켜기
```

`optional = true`인 의존성은 그 feature를 켜야만 실제로 빌드된다.
같은 코드베이스로 **로봇용(가벼움)과 개발용(시각화 포함)을 나누는 데** 쓴다.

---

# 11. 전체를 한 장에

```toml
# ─── 루트 Cargo.toml ───────────────────────────
[workspace]
resolver = "2"
members = ["src/interfaces", "src/fusion"]

[workspace.package]
version = "0.1.0"
edition = "2021"

[workspace.dependencies]
nalgebra = "0.33"

[profile.release]
opt-level = 3
lto = true

# ─── src/fusion/Cargo.toml ─────────────────────
[package]
name = "slam_fusion"
version.workspace = true
edition.workspace = true

[lib]
path = "lib.rs"

[dependencies]
nalgebra.workspace = true
slam_interfaces = { path = "../interfaces" }

[dev-dependencies]
approx = "0.5"
```

---

# 12. Mini Practice

Chapter 1에서 만든 `/tmp/rust_ws`에 이어서 해본다.

```bash
cd /tmp/rust_ws

# 1) 의존성을 직접 추가해 본다
cargo add --package demo_fusion serde --features derive
cat src/fusion/Cargo.toml          # [dependencies] 에 추가된 것 확인

# 2) workspace 상속으로 바꿔 본다
#    루트 Cargo.toml 에 추가:
#      [workspace.dependencies]
#      serde = { version = "1", features = ["derive"] }
#    src/fusion/Cargo.toml 을 수정:
#      serde.workspace = true
cargo check

# 3) Cargo 가 최종적으로 해석한 값 확인
cargo metadata --no-deps --format-version 1 \
  | python3 -c "import sys,json; [print(p['name'], p['version'], p['edition']) for p in json.load(sys.stdin)['packages']]"

# 4) binary 를 관례 위치에 두면 — 동작한다
mkdir -p src/fusion/src/bin
echo 'fn main(){ println!("convention works"); }' > src/fusion/src/bin/demo.rs
cargo run -p demo_fusion --bin demo

# 5) 관례를 벗어난 위치로 옮기면 — 실패한다
rm -rf src/fusion/src
mkdir -p src/fusion/tools
echo 'fn main(){ println!("hi"); }' > src/fusion/tools/demo.rs
cargo run -p demo_fusion --bin demo
#   error: no bin target named `demo` in `demo_fusion` package

# 6) [[bin]] 으로 명시하면 다시 동작한다
cat >> src/fusion/Cargo.toml <<'TOML'

[[bin]]
name = "demo"
path = "tools/demo.rs"
TOML
cargo run -p demo_fusion --bin demo
```

4·5·6번이 §6의 내용이다. **관례 위치면 설정이 필요 없고, 벗어나면 `path`로 알려줘야 한다.**

---

# 13. 오늘의 핵심

```text
  루트 Cargo.toml                    하위 Cargo.toml
  ────────────────────────           ──────────────────────────
  [workspace]                        [package]
    members    누구를 묶나              name      crate 이름이 된다
    exclude                            version.workspace = true
                                                   ↑ 루트에서 상속
  [workspace.package]
    version   ─────────────────▶     [lib]      library crate
    edition   ─────────────────▶       path, crate-type

  [workspace.dependencies]           [[bin]]    binary crate (여러 개)
    serde     ─────────────────▶
                                     [dependencies]
  [profile.release]                    serde.workspace = true
    루트에만 쓸 수 있다                  ../path, crates.io, git
```

---

# 14. 반드시 구분할 것

```text
[workspace.dependencies] 에 적음  ≠  실제로 쓰임
   하위에서 serde.workspace = true 로 명시해야 적용된다

[dependencies]  ≠  [dev-dependencies]
   후자는 최종 산출물에 들어가지 않는다

[lib]  ≠  [[bin]]
   대괄호 1개 = 최대 하나 / 2개 = 여러 개

crate-type = "rlib"  ≠  "cdylib"
   rlib 은 Rust 전용. C++ 에서 부르려면 cdylib / staticlib

[profile] 은 루트에만
   하위에 적으면 무시된다

git 의존성의 branch  ≠  rev
   branch 는 움직인다. 재현하려면 rev 나 tag

관례를 따름  vs  path 로 지정
   따를 수 있으면 따르는 쪽이 설정이 없어서 낫다
```

---

# 15. Chapter 연결

```text
Chapter 1
workspace / package / crate / module — 지도

Chapter 2  ← 여기
Cargo.toml — 그 지도를 적어 놓은 설정 파일

Chapter 3
cargo 명령어와 target/ — 설정대로 빌드하면 무엇이 생기나

Chapter 4
의존성과 Cargo.lock — 버전이 실제로 어떻게 정해지나
```
