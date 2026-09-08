---
title: "Chapter 3. cargo 명령어와 target/"
importance: 4
---

> **Goal:** `cargo check` / `build` / `run` / `test`가 각각 무엇을 하는지 구분하고,
> `target/` 아래에 무엇이 생기는지, debug와 release가 왜 그렇게 차이 나는지 안다.

Chapter 2까지가 "무엇을 만들지 적는 법"이었다면
이 chapter는 "그래서 실제로 무엇이 만들어지는가"다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo check` · `cargo build --release` · `cargo test`

| 명령어                          | 하는 일                                                |
| :------------------------------ | :----------------------------------------------------- |
| `cargo check`                   | 타입·문법만 검사. **실행파일을 안 만들어서 훨씬 빠름** |
| `cargo build`                   | debug 빌드 → `target/debug/`                           |
| `cargo build --release`         | 최적화 빌드 → `target/release/`                        |
| `cargo run`                     | 빌드하고 바로 실행                                     |
| `cargo run --release -- --arg1` | `--` 뒤는 프로그램에 넘기는 인자                       |
| `cargo test`                    | 테스트 빌드 후 실행                                    |
| `cargo clean`                   | `target/` 통째로 삭제                                  |
| `cargo build -p <pkg>`          | workspace 안 특정 package만                            |
| `cargo build --workspace`       | 전체 member                                            |
| `cargo build -v`                | 실제 `rustc` 호출을 보여줌                             |
| `cargo build --timings`         | 무엇이 오래 걸리는지 HTML 리포트                       |
| `cargo fmt`                     | 코드 자동 정렬                                         |
| `cargo clippy`                  | 린트. 관용적이지 않은 코드 지적                        |
| `cargo doc --open`              | 문서 생성 후 브라우저로                                |

---

# 1. `check` / `build` / `run` — 어디까지 하나

가장 많이 쓰는 세 개인데, 하는 일의 범위가 다르다.

```text
              파싱   타입검사   코드생성   링크   실행
cargo check    ✓       ✓        ✗       ✗     ✗
cargo build    ✓       ✓        ✓       ✓     ✗
cargo run      ✓       ✓        ✓       ✓     ✓
```

`check`가 코드 생성과 링크를 건너뛴다는 것이 핵심이다.
컴파일 시간의 상당 부분이 그 두 단계라서, 개발 중에는 `check`가 훨씬 빠르다.

```text
코드를 고치는 중        cargo check    ← 문법·타입 오류만 빨리 확인
돌려봐야 할 때          cargo run
로봇에 올릴 때          cargo build --release
```

에디터의 rust-analyzer가 배경에서 계속 돌리는 것도 `check`에 해당한다.

---

# 2. `target/` 안에 무엇이 생기나

```text
target/
├── debug/
│   ├── slam-node              ← 실행파일
│   ├── libslam_interfaces.rlib ← library crate 산출물
│   ├── deps/                  ← 의존 crate 들의 산출물
│   ├── build/                 ← build.rs 결과
│   └── incremental/           ← 증분 컴파일 캐시
├── release/
│   └── (같은 구조, 최적화 버전)
└── CACHEDIR.TAG
```

몇 가지 알아둘 점.

```text
· target/ 은 통째로 재생성 가능하다  →  .gitignore 에 넣는다
· workspace 면 루트에 하나만 생긴다  →  member 들이 공유
· 금방 GB 단위가 된다             →  cargo clean 으로 정리
```

`.gitignore`는 보통 이 정도다.

```gitignore
/target
```

`Cargo.lock`은 상황에 따라 다르다. (Chapter 4 §6)

---

# 3. workspace에서는 `target/`을 공유한다

```text
SeoulDynamics.SLAM/
├── Cargo.toml
├── target/            ← 여기 하나만 생긴다
└── src/
    ├── interfaces/    ← 각자 target/ 을 만들지 않는다
    └── fusion/
```

이게 workspace의 실질적인 이득 중 하나다.
`interfaces`와 `fusion`이 둘 다 `nalgebra`를 쓴다면,
`nalgebra`는 **한 번만 컴파일된다.**

package를 따로따로 두면 각자 컴파일해서 시간과 디스크를 두 배로 쓴다.

---

# 4. 빌드가 오래 걸리는 이유와 증분 빌드

Rust는 첫 빌드가 느리다. 의존성을 전부 소스에서 컴파일하기 때문이다.

```text
첫 빌드          의존성 전부 컴파일       분 단위
그 다음부터       바뀐 crate 만 다시       초 단위
```

Chapter 1 §7에서 본 것이 여기서 효과를 낸다.

```text
crate 하나에 파일 20개
   → 파일 하나만 고쳐도 crate 전체를 다시 컴파일

crate 5개로 나눔
   → 고친 crate 만 다시 컴파일. 나머지 4개는 캐시
```

**crate를 적절히 나누면 개발 중 빌드 시간이 눈에 띄게 줄어든다.**

무엇이 오래 걸리는지 궁금하면 측정하면 된다.

```bash
cargo build --timings
# target/cargo-timings/cargo-timing.html 이 생긴다
```

---

# 5. debug와 release는 얼마나 다른가

```toml
[profile.dev]        # cargo build
opt-level = 0
debug = true

[profile.release]    # cargo build --release
opt-level = 3
debug = false
```

차이가 생각보다 크다.

```text
                debug              release
빌드 시간        빠름               느림 (2~5배)
실행 속도        느림               빠름
                (연산 중심 코드는 10~50배 차이도 난다)
바이너리 크기     큼 (심볼 포함)      작음
디버거           심볼 다 있음        기본적으로 없음
```

**로봇에 올리는 것은 반드시 `--release`다.**
`cargo run`만 하다가 "Rust가 C++보다 느리다"고 결론 내리는 실수가 흔한데,
거의 항상 debug 빌드로 재본 것이다.

성능을 재려면:

```bash
cargo build --release
./target/release/slam-node
```

배포판인데 디버깅도 하고 싶으면 절충한다.

```toml
[profile.release]
debug = true       # 심볼만 남긴다. 최적화는 그대로
```

---

# 6. `cargo test`

테스트는 별도 설정 없이 코드 안에 같이 쓴다.

```rust
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        assert_eq!(add(2, 3), 5);
    }
}
```

```bash
cargo test                      # 전체
cargo test -p slam_fusion       # 특정 package
cargo test add_works            # 이름으로 필터
cargo test -- --nocapture       # println! 출력 보기
cargo test --release            # 최적화 상태로
```

`#[cfg(test)]`는 **테스트 빌드일 때만 컴파일하라**는 뜻이다.
그래서 테스트 코드가 배포 바이너리에 들어가지 않는다.

`cargo test`는 기본적으로 출력을 삼킨다.
`println!`이 안 보이면 `-- --nocapture`를 붙인다. `--`가 두 번인 것에 주의한다.

```text
cargo test -- --nocapture
           │  └── 테스트 실행기에게 주는 인자
           └──── 여기까지가 cargo 인자
```

---

# 7. `fmt`와 `clippy`

```bash
cargo fmt                  # 코드 정렬 (파일을 고침)
cargo fmt --check          # 고치지 않고 검사만 (CI 용)

cargo clippy               # 린트
cargo clippy --fix         # 자동으로 고칠 수 있는 것은 고침
cargo clippy -- -D warnings  # 경고를 에러로 (CI 용)
```

`fmt`는 **취향 논쟁을 없애는 도구**다. 팀에서 스타일을 정할 필요가 없어진다.
`clippy`는 컴파일은 되지만 관용적이지 않은 코드를 지적한다.

```text
rustc      틀린 코드를 잡는다
clippy     맞지만 어색한 코드를 잡는다
fmt        모양을 맞춘다
```

CI에서는 보통 이 셋을 다 돌린다.

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

설치가 안 되어 있으면:

```bash
rustup component add rustfmt clippy
```

---

# 8. 툴체인 — `rustup`과 `cargo`의 차이

```text
rustup     Rust 자체(컴파일러·툴체인)를 설치하고 버전을 관리
cargo      프로젝트를 빌드하고 의존성을 관리
```

```bash
rustup show                    # 지금 쓰는 툴체인
rustup update                  # Rust 업데이트
rustup default stable
rustup target add aarch64-unknown-linux-gnu   # 크로스 컴파일 대상 추가
```

프로젝트에 버전을 고정하고 싶으면 `rust-toolchain.toml`을 둔다.

```toml
[toolchain]
channel = "1.82.0"
components = ["rustfmt", "clippy"]
targets = ["aarch64-unknown-linux-gnu"]
```

이 파일이 있으면 **누가 어디서 빌드하든 같은 컴파일러를 쓴다.**
Jetson 배포처럼 재현성이 중요한 곳에서는 넣어두는 편이 낫다.

---

# 9. 크로스 컴파일 — x86에서 Jetson용 만들기

Edge Computing Chapter 2의 아키텍처 이야기가 그대로 이어진다.

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

산출물 위치가 달라진다.

```text
target/aarch64-unknown-linux-gnu/release/slam-node
```

순수 Rust 코드는 이걸로 되지만, C 라이브러리에 링크하는 crate가 있으면
**해당 아키텍처용 링커와 라이브러리**가 추가로 필요하다.
그래서 실무에서는 Jetson용 Docker 컨테이너 안에서 그냥 빌드하는 쪽이 간단할 때가 많다. (Chapter 6)

확인:

```bash
file target/aarch64-unknown-linux-gnu/release/slam-node
# ELF 64-bit LSB ... ARM aarch64
```

---

# 10. Mini Practice

```bash
cd /tmp/rust_ws

# 1) check 와 build 의 시간 차이
cargo clean
time cargo check
cargo clean
time cargo build

# 2) 무엇이 생겼는지
find target/debug -maxdepth 1 -type f | head
du -sh target

# 3) release 와 크기 비교
cargo build --release
ls -la target/debug/demo target/release/demo

# 4) rustc 를 몇 번 부르는지
cargo clean && cargo build -v 2>&1 | grep -c "Running.*rustc"

# 5) 무엇이 오래 걸렸나
cargo clean && cargo build --timings
# target/cargo-timings/cargo-timing.html 열기
```

4번을 해보면 숫자가 package 개수보다 **크게** 나온다.

```text
$ cargo build -v | grep -c "Running.*rustc"
14
$ cargo metadata --format-version 1 | ... (package 수)
9
```

Chapter 1 §7의 "crate 하나 = rustc 한 번"은 맞지만,
`rustc` 호출 수가 package 수와 같지는 않다. 차이는 두 가지에서 온다.

```text
· proc-macro crate 는 두 번 컴파일된다
    컴파일러 안에서 돌아야 하므로 호스트용으로 한 번,
    최종 산출물용으로 한 번
· build.rs 가 있으면 그것도 별도로 컴파일하고 실행한다
```

`serde`처럼 `derive` feature를 켜면 `serde_derive`가 proc-macro라서 이런 일이 생긴다.
**의존성 하나를 추가했을 뿐인데 빌드가 눈에 띄게 느려지는** 이유이기도 하다.

---

# 11. 오늘의 핵심

```text
   소스                     Cargo                     target/

  Cargo.toml  ──┐
                ├──▶  cargo check    타입만 검사   (산출물 없음)
  lib.rs      ──┤
  main.rs     ──┤──▶  cargo build    debug        target/debug/
                │──▶  cargo build    release      target/release/
                │       --release                  ↑ 10~50배 빠름
                └──▶  cargo test     테스트 포함    target/debug/deps/

          workspace 면 target/ 은 루트에 하나. member 들이 공유한다
```

---

# 12. 반드시 구분할 것

```text
cargo check  ≠  cargo build
   check 는 코드 생성·링크를 건너뛴다. 훨씬 빠르다

debug  ≠  release
   성능 측정은 반드시 --release. 10~50배 차이

rustup  ≠  cargo
   rustup 은 컴파일러를 관리, cargo 는 프로젝트를 관리

cargo test -- --nocapture
   앞의 -- 까지가 cargo, 뒤는 테스트 실행기 인자

rustc 가 잡는 것  ≠  clippy 가 잡는 것
   틀린 코드 vs 어색한 코드

target/ 은 산출물
   .gitignore 에 넣는다. 언제든 재생성 가능

--target 지정  ≠  그냥 빌드
   경로가 target/<triple>/release/ 로 바뀐다
```

---

# 13. Chapter 연결

```text
Chapter 2
Cargo.toml — 무엇을 만들지 적는다

Chapter 3  ← 여기
cargo — 적은 대로 만든다. target/ 에 무엇이 생기나

Chapter 4
의존성과 Cargo.lock — 버전은 누가 정하나

Edge Computing Chapter 2
ARM vs x86 — 크로스 컴파일이 필요한 이유
```
