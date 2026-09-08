---
title: "Chapter 4. 의존성과 Cargo.lock"
importance: 5
---

> **Goal:** `"1.0"`이라고 적었을 때 실제로 어떤 버전이 설치되는지 설명하고,
> `Cargo.lock`을 커밋할지 말지를 근거를 가지고 결정할 수 있다.

Chapter 2에서 `[dependencies]`에 한 줄 적는 법을 봤다.
그런데 `serde = "1.0"`이라고 적으면 정확히 뭐가 설치될까?
`1.0.0`일까, 최신일까? 그리고 내일 다시 빌드하면 같은 게 나올까?

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo tree` · `cargo update` · `cargo add`

| 명령어                                    | 하는 일                                          |
| :---------------------------------------- | :----------------------------------------------- |
| `cargo add serde`                         | 의존성 추가 (`Cargo.toml` 자동 수정)             |
| `cargo add serde@1.0.200`                 | 버전 지정해서 추가                               |
| `cargo remove serde`                      | 제거                                             |
| `cargo tree`                              | 의존 트리 전체                                   |
| `cargo tree -d`                           | **중복된 버전** 찾기 (같은 crate의 여러 버전)    |
| `cargo tree -i serde`                     | 누가 이 crate를 끌어왔는지 역추적                |
| `cargo update`                            | `Cargo.lock`의 버전을 최신으로                   |
| `cargo update -p serde`                   | 특정 crate만                                     |
| `cargo update -p serde --precise 1.0.190` | 특정 버전으로 고정                               |
| `cargo fetch`                             | 의존성 내려받기만 (오프라인 준비)                |
| `cargo build --offline`                   | 네트워크 없이 빌드                               |
| `cargo audit`                             | 알려진 취약점 검사 (`cargo install cargo-audit`) |
| `cargo outdated`                          | 오래된 의존성 (`cargo install cargo-outdated`)   |

---

# 1. `"1.0"`은 정확한 버전이 아니라 **범위**다

가장 먼저 고칠 오해가 이것이다.

```toml
serde = "1.0"
```

이건 "1.0.0을 써라"가 아니다. **"1.0.0 이상, 2.0.0 미만이면 아무거나"**라는 뜻이다.

```text
"1.0"      →  >=1.0.0, <2.0.0      실제로는 1.0.229 가 설치될 수 있다
"1.0.190"  →  >=1.0.190, <2.0.0    ← 이것도 범위다!
"=1.0.190" →  정확히 1.0.190       고정하려면 = 를 붙인다
"^1.0"     →  "1.0" 과 같음 (기본값)
"~1.0.5"   →  >=1.0.5, <1.1.0      마지막 자리만 열어둠
"*"        →  아무거나             쓰지 말 것
```

**`serde = "1.0.190"`이라고 적어도 `1.0.229`가 설치된다**는 것이 핵심이다.
Cargo는 그걸 "최소 1.0.190은 되어야 한다"로 읽는다.

---

# 2. Semantic Versioning — 왜 그런 규칙인가

```text
    1  .  2  .  3
    │     │     └── PATCH   버그 수정. 쓰는 쪽 코드는 안 바뀐다
    │     └──────── MINOR   기능 추가. 기존 코드는 그대로 동작한다
    └────────────── MAJOR   호환성 깨짐. 코드를 고쳐야 한다
```

Cargo가 `"1.0"`을 `<2.0.0`으로 읽는 이유가 이것이다.
**MAJOR가 같으면 호환된다는 약속**이므로, 그 안에서는 자유롭게 올려도 안전하다는 전제다.

0.x는 예외적으로 취급한다.

```text
"0.3"    →  >=0.3.0, <0.4.0     ← 0.x 에서는 MINOR 가 MAJOR 역할
"0.0.3"  →  정확히 0.0.3
```

`nalgebra = "0.33"`이 `0.34`로 안 올라가는 이유다.
0.x 라이브러리는 아직 API가 바뀔 수 있다고 보는 것이다.

---

# 3. 그러면 실제 버전은 누가 정하나 — `Cargo.lock`

`Cargo.toml`은 범위를 적고, **`Cargo.lock`이 실제 버전을 적는다.**

```text
Cargo.toml                    Cargo.lock
──────────────────           ────────────────────────
serde = "1.0"        ──▶     name = "serde"
   "범위"                     version = "1.0.229"
                             checksum = "a1b2c3..."
                                "정확한 하나"
```

```toml
# Cargo.lock (자동 생성. 손으로 고치지 않는다)
[[package]]
name = "serde"
version = "1.0.229"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "5c3e9b2a..."
```

동작은 이렇다.

```text
Cargo.lock 이 없다   →  범위 안에서 최신을 고르고, lock 파일을 만든다
Cargo.lock 이 있다   →  적힌 버전을 그대로 쓴다. 최신을 찾지 않는다
```

**그래서 lock 파일이 있으면 오늘과 내일의 빌드가 같다.**
`cargo build`를 백 번 해도 버전은 안 움직인다. 움직이려면 명시적으로 말해야 한다.

```bash
cargo update              # 범위 안에서 최신으로 갱신
cargo update -p serde     # serde 만
```

---

# 4. `cargo tree` — 실제로 뭐가 끌려왔는지

의존성은 전염된다. 하나 추가하면 그게 끌고 오는 것들이 딸려온다.

```bash
$ cargo tree
demo_fusion v0.1.0
├── demo_interfaces v0.1.0
└── serde v1.0.229
    ├── serde_core v1.0.229
    └── serde_derive v1.0.229 (proc-macro)
        ├── proc-macro2 v1.0.107
        │   └── unicode-ident v1.0.24
        ├── quote v1.0.47
        └── syn v3.0.5
```

`serde` 하나를 넣었는데 7개가 딸려왔다.
Chapter 3 §10에서 `rustc` 호출이 package 수보다 많았던 이유가 여기 보인다.

역추적도 된다.

```bash
cargo tree -i syn        # syn 을 누가 끌어왔나
```

---

# 5. 같은 crate의 여러 버전이 동시에 들어올 수 있다

이게 처음 보면 놀랍다.

```bash
$ cargo tree -d
rand v0.7.3
└── (A 가 씀)

rand v0.8.5
└── (B 가 씀)
```

`rand` 0.7과 0.8은 MAJOR가 다르므로 (0.x에서는 MINOR가 MAJOR 역할)
Cargo는 **호환되지 않는다고 보고 둘 다 넣는다.**

```text
결과
  · 빌드는 된다
  · 바이너리가 커진다
  · A 가 만든 rand 0.7 타입을 B 에 넘기면  →  타입 에러
       "expected rand::Rng, found rand::Rng"  ← 같아 보이는데 다르다
```

이 에러 메시지가 처음엔 미친 소리처럼 보인다.
`cargo tree -d`로 중복을 확인하는 습관이 그래서 필요하다.

해결은 보통 **버전을 맞추는 것**이다. `[workspace.dependencies]`로 한곳에서 관리하면
애초에 이런 일이 잘 안 생긴다. (Chapter 2 §4)

---

# 6. `Cargo.lock`을 커밋할까?

정답이 갈리는 것처럼 알려져 있지만, 기준은 명확하다.

```text
커밋한다        애플리케이션 / 최종 산출물
                (binary crate, 로봇에 올리는 프로그램)
                → 모두가 정확히 같은 것을 빌드해야 한다

커밋 안 한다     남에게 라이브러리로 배포하는 crate
                → 쓰는 쪽의 버전 선택을 방해하지 않는다
```

**로봇 프로젝트는 거의 항상 전자다.** 커밋한다.

```text
Cargo.lock 을 커밋하면
  ✓ 팀원 모두 같은 버전으로 빌드
  ✓ CI 와 로컬이 일치
  ✓ 지난달 빌드를 지금 재현 가능
  ✓ 어제까지 되던 게 오늘 깨지는 일이 없음
```

`Cargo.lock`을 안 올렸을 때의 전형적인 사고는 이렇다.

```text
내 노트북      빌드 됨   (serde 1.0.190 을 잠금)
CI             빌드 깨짐 (serde 1.0.229 를 새로 받음)
               "내 컴퓨터에선 되는데요"
```

---

# 7. 오프라인 빌드 — 로봇 현장에서

현장에는 인터넷이 없을 수 있다. 미리 받아두면 된다.

```bash
# 인터넷 있을 때
cargo fetch

# 현장에서
cargo build --offline --release
```

Docker 이미지에 굽는다면 `cargo fetch`를 Dockerfile 단계에 넣는다.
Edge Computing Chapter 9.5의 "이미지에 굽는 것 vs 마운트하는 것" 기준 그대로다.

```dockerfile
# 의존성만 먼저 받아서 레이어 캐시를 살린다
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main(){}" > src/main.rs && cargo fetch
COPY . .
RUN cargo build --release --offline
```

의존성이 안 바뀌면 그 레이어를 다시 안 받는다.

---

# 8. `vendor` — 의존성을 저장소에 넣기

더 강하게 고정하려면 소스를 통째로 가져온다.

```bash
cargo vendor > .cargo/config.toml
```

```text
vendor/
├── serde/
├── syn/
└── ...
```

```text
장점    네트워크 없이 완전히 재현 가능. 원격 레지스트리가 사라져도 안전
단점    저장소가 커진다 (수백 MB). 업데이트가 번거롭다
```

방산·산업용처럼 재현성 요구가 강한 곳에서 쓴다.
보통은 `Cargo.lock` 커밋 + `cargo fetch`로 충분하다.

---

# 9. 의존성을 늘릴 때 따져볼 것

Rust는 `cargo add` 한 줄이면 되니까 의존성이 쉽게 불어난다.

```text
crate 를 하나 추가하기 전에

· 이게 끌고 오는 게 몇 개인가?        cargo tree 로 미리 본다
· 빌드 시간이 얼마나 늘어나나?         proc-macro 면 특히
· 마지막 릴리스가 언제인가?            방치된 crate 인가
· 라이선스가 맞나?                    상용 제품이면 중요
· 표준 라이브러리로 5줄이면 되는 일인가?
```

`cargo audit`으로 알려진 취약점도 확인할 수 있다.

```bash
cargo install cargo-audit
cargo audit
```

---

# 10. Mini Practice

```bash
cd /tmp/rust_ws

# 1) serde 를 넣고 무엇이 딸려오는지 본다
cargo add -p demo_fusion serde --features derive
cargo tree

# 2) Cargo.lock 에 어떤 버전이 박혔나
grep -A2 'name = "serde"' Cargo.lock

# 3) Cargo.toml 은 범위, Cargo.lock 은 확정임을 확인
grep serde src/fusion/Cargo.toml     # "1.0" 같은 범위
grep -A1 'name = "serde"' Cargo.lock # 1.0.xxx 같은 확정 버전

# 4) lock 을 지우면 다시 고른다
cp Cargo.lock /tmp/lock.bak
rm Cargo.lock
cargo check                          # 새로 생성됨
diff <(grep version /tmp/lock.bak) <(grep version Cargo.lock)

# 5) 중복 버전 확인
cargo tree -d                        # 지금은 아마 비어 있다

# 6) 누가 syn 을 끌어왔나
cargo tree -i syn
```

3번에서 **`Cargo.toml`의 문자열과 `Cargo.lock`의 버전이 다르다**는 것을 눈으로 본다.
이 chapter의 핵심이다.

---

# 11. 오늘의 핵심

```text
   Cargo.toml                          Cargo.lock
   ──────────────────────             ────────────────────────
   내가 손으로 쓴다                     Cargo 가 만든다
   serde = "1.0"                       version = "1.0.229"
   "범위"                              "정확히 하나"
   >=1.0.0, <2.0.0                     checksum 까지 고정

          │                                    │
          │  Cargo.lock 이 없으면 ──────────────┘
          │     범위 안에서 최신을 골라 lock 을 만든다
          │
          └─ Cargo.lock 이 있으면
                적힌 버전 그대로. cargo update 를 해야 움직인다

        로봇 프로젝트 → Cargo.lock 을 커밋한다
```

---

# 12. 반드시 구분할 것

```text
"1.0.190"  ≠  정확히 1.0.190
   범위다. 고정하려면 "=1.0.190"

Cargo.toml  ≠  Cargo.lock
   범위 선언 vs 확정된 버전

cargo build  ≠  cargo update
   build 는 lock 을 존중한다. update 만 버전을 움직인다

0.3  vs  1.3
   0.x 는 MINOR 가 MAJOR 역할. 0.3 → 0.4 는 호환 안 됨

같은 crate 의 두 버전
   동시에 들어올 수 있다. 타입은 서로 호환되지 않는다
   cargo tree -d 로 확인

Cargo.lock 커밋
   애플리케이션 → 한다 / 배포용 라이브러리 → 안 한다

cargo fetch  ≠  cargo build
   fetch 는 받기만. --offline 빌드를 위한 준비
```

---

# 13. Chapter 연결

```text
Chapter 2
Cargo.toml — [dependencies] 를 적는 법

Chapter 3
cargo build — 의존성이 컴파일 시간을 늘리는 이유

Chapter 4  ← 여기
버전은 누가 어떻게 정하나

Chapter 5
module — 이제 crate 안으로 들어간다

Edge Computing Chapter 9.5
Docker 레이어 캐시와 cargo fetch
```
