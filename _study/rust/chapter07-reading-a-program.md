---
title: "Chapter 7. 돌아가는 프로그램 하나 읽기"
importance: 8
---

> **Goal:** 실제로 컴파일되고 실행되는 Rust 프로그램 하나를 처음부터 끝까지 읽으면서,
> 거기 쓰인 문법을 나오는 순서대로 이해한다.

문법을 하나씩 배우고 예제를 보는 순서가 아니다.
**먼저 돌아가는 프로그램을 두고, 그 안에 실제로 쓰인 것만 설명한다.**
안 쓰인 문법은 안 나온다. 필요해지면 그때 나온다.

이 chapter의 프로그램은 IMU 로그를 읽어 통계를 내고 샘플 누락을 찾는다.
Chapter 8, 9에서 **같은 프로그램을 계속 고쳐 가며** 소유권과 에러 처리를 배운다.

> 참고: [Rust 프로그래밍 언어 (한국어판)](https://doc.rust-kr.org/title-page.html).
> 각 절 끝에 대응하는 장을 달아 두었다.

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo run` · `cargo check`

| 명령어                  | 하는 일                        |
| :---------------------- | :----------------------------- |
| `cargo new imu_stats`   | 프로젝트 생성 (binary)         |
| `cargo run`             | 빌드하고 실행                  |
| `cargo run -q`          | 빌드 로그 없이 프로그램 출력만 |
| `cargo check`           | 문법·타입만 빠르게 확인        |
| `cargo fmt`             | 코드 정렬                      |
| `cargo clippy`          | 더 나은 표현 제안              |
| `rustc --explain E0382` | **에러 코드 설명 보기**        |

---

# 1. 프로그램 전체

먼저 통째로 본다. 지금은 이해 안 되는 줄이 있어도 괜찮다.

```rust
// src/main.rs
#[derive(Debug, Clone, Copy)]
struct Sample {
    stamp: f64,
    ax: f64,
    ay: f64,
    az: f64,
}

impl Sample {
    fn magnitude(&self) -> f64 {
        (self.ax * self.ax + self.ay * self.ay + self.az * self.az).sqrt()
    }
}

fn main() {
    let raw = "\
0.000,0.01,-0.02,9.79
0.005,0.02,-0.01,9.81
0.010,0.15,0.30,10.90
0.015,0.01,0.00,9.80
0.045,-0.02,0.01,9.78";

    let mut samples = Vec::new();

    for line in raw.lines() {
        let cols: Vec<&str> = line.split(',').collect();
        let s = Sample {
            stamp: cols[0].parse().unwrap(),
            ax: cols[1].parse().unwrap(),
            ay: cols[2].parse().unwrap(),
            az: cols[3].parse().unwrap(),
        };
        samples.push(s);
    }

    println!("샘플 {}개", samples.len());

    let mut sum = 0.0;
    for s in &samples {
        sum += s.magnitude();
    }
    let mean = sum / samples.len() as f64;
    println!("평균 가속도 크기: {:.3} m/s^2", mean);

    let mut worst = &samples[0];
    for s in &samples {
        if s.magnitude() > worst.magnitude() {
            worst = s;
        }
    }
    println!("최대: {:.3} at t={:.3}", worst.magnitude(), worst.stamp);

    let expected_dt = 0.005;
    for pair in samples.windows(2) {
        let dt = pair[1].stamp - pair[0].stamp;
        if dt > expected_dt * 1.5 {
            println!("간격 이상: t={:.3} 에서 {:.3}s (기대 {:.3}s)",
                     pair[0].stamp, dt, expected_dt);
        }
    }
}
```

직접 돌려 본다.

```bash
cargo new imu_stats
cd imu_stats
# 위 코드를 src/main.rs 에 붙여넣고
cargo run
```

```text
샘플 5개
평균 가속도 크기: 10.017 m/s^2
최대: 10.905 at t=0.010
간격 이상: t=0.015 에서 0.030s (기대 0.005s)
```

마지막 줄이 이 프로그램의 쓸모다. **0.005초 간격이어야 하는데 0.030초가 비었다.**
Edge Computing Chapter 11의 시간 동기화 문제를 로그에서 찾아내는 일이다.

이제 한 조각씩 읽는다.

---

# 2. `struct` — 데이터를 묶기

```rust
struct Sample {
    stamp: f64,
    ax: f64,
    ay: f64,
    az: f64,
}
```

C의 구조체와 거의 같다. 관련된 값을 하나로 묶는다.

```text
struct 이름 {
    필드이름: 타입,
    ...
}
```

`f64`는 64비트 부동소수점이다. Rust는 타입 이름에 **크기가 그대로 들어간다.**

```text
i8  i16  i32  i64  i128     부호 있는 정수
u8  u16  u32  u64  u128     부호 없는 정수
f32 f64                     부동소수점
usize isize                 포인터 크기 (배열 인덱스에 쓴다)
bool  char                  참거짓, 유니코드 문자 하나
```

C의 `int`, `long`처럼 **플랫폼마다 크기가 달라지는 타입이 없다.**
x86에서 만든 코드가 Jetson에서 다르게 도는 일이 줄어든다.

만드는 방법은 이렇다.

```rust
let s = Sample {
    stamp: 0.0,
    ax: 0.1,
    ay: 0.0,
    az: 9.81,
};
```

**모든 필드를 반드시 채워야 한다.** 하나라도 빠지면 컴파일 에러다.
C처럼 초기화 안 한 필드가 쓰레기 값으로 남는 일이 없다.

> rust-kr [5.1 구조체 정의 및 인스턴트화](https://doc.rust-kr.org/ch05-01-defining-structs.html),
> [3.2 데이터 타입](https://doc.rust-kr.org/ch03-02-data-types.html)

---

# 3. `#[derive(...)]` — 공짜로 얻는 기능

```rust
#[derive(Debug, Clone, Copy)]
struct Sample { ... }
```

`#[...]`는 어트리뷰트(attribute), 컴파일러에게 주는 지시다.
`derive`는 **흔한 기능을 자동으로 구현해 달라**는 뜻이다.

```text
Debug    {:?} 로 출력할 수 있게 된다
Clone    .clone() 으로 복사본을 만들 수 있게 된다
Copy     대입할 때 자동으로 복사된다 (Chapter 8 의 핵심)
```

`Debug`가 붙으면 이게 된다.

```rust
println!("{:?}", s);
// Sample { stamp: 0.0, ax: 0.1, ay: 0.0, az: 9.81 }

println!("{:#?}", s);   // 여러 줄로 예쁘게
```

**디버깅할 때 거의 항상 붙인다.** `Debug`가 없으면 `{:?}`가 컴파일 에러다.

`Copy`는 지금은 "작고 단순한 값이라 복사가 싸다"는 표시 정도로 넘어간다.
Chapter 8에서 이게 왜 중요한지 나온다.

> rust-kr [부록 C - 파생 가능한 트레이트](https://doc.rust-kr.org/appendix-03-derivable-traits.html)

---

# 4. `impl` — 구조체에 동작 붙이기

```rust
impl Sample {
    fn magnitude(&self) -> f64 {
        (self.ax * self.ax + self.ay * self.ay + self.az * self.az).sqrt()
    }
}
```

`impl`은 implementation의 줄임말이다. **데이터(struct)와 동작(impl)을 따로 쓴다.**

```text
C++         class 안에 필드와 메서드가 같이 있다
Rust        struct 에 필드,  impl 블록에 메서드
```

`&self`가 첫 인자인 것이 메서드의 표시다.

```text
fn magnitude(&self) -> f64
              │        └── 반환 타입
              └── "이 Sample 을 빌려서 읽기만 하겠다"
```

`&self`의 세 가지 형태가 있고, 이것이 Chapter 8의 예고편이다.

```text
&self       읽기만 한다        가장 흔하다
&mut self   고칠 수 있다
self        가져가 버린다      호출 후 원본을 못 쓴다
```

호출은 점으로 한다.

```rust
let m = s.magnitude();
```

마지막 줄에 세미콜론이 없는 것에 주의한다.
**Rust에서 세미콜론 없는 마지막 식이 반환값이다.**

```rust
fn double(x: f64) -> f64 {
    x * 2.0        // return 을 안 써도 된다
}

fn double2(x: f64) -> f64 {
    return x * 2.0;   // 이렇게 써도 되지만 관용적이지 않다
}
```

> rust-kr [5.3 메서드 문법](https://doc.rust-kr.org/ch05-03-method-syntax.html),
> [3.3 함수](https://doc.rust-kr.org/ch03-03-how-functions-work.html)

---

# 5. `let`과 `mut` — 기본이 불변

```rust
let raw = "...";              // 못 바꾼다
let mut samples = Vec::new(); // 바꿀 수 있다
let mut sum = 0.0;
```

**Rust는 아무것도 안 붙이면 변수를 못 바꾼다.** 다른 언어와 반대다.

```rust
let x = 5;
x = 6;        // error[E0384]: cannot assign twice to immutable variable
```

바꾸려면 명시해야 한다.

```rust
let mut x = 5;
x = 6;        // OK
```

귀찮아 보이지만 이득이 있다.

```text
· "이 값은 안 바뀐다"가 코드에 적혀 있다
· 실수로 덮어쓰는 버그를 컴파일러가 잡는다
· 동시성에서 안전한 것과 아닌 것이 구분된다 (rust-kr 16장)
```

`mut`을 안 붙였는데 필요하면 컴파일러가 알려준다.

```text
help: consider changing this to be mutable: `mut samples`
```

**Rust 에러 메시지는 대부분 고치는 방법까지 알려준다.** 읽는 습관을 들이는 게 좋다.

> rust-kr [3.1 변수와 가변성](https://doc.rust-kr.org/ch03-01-variables-and-mutability.html)

---

# 6. 타입을 안 적어도 되는 이유

```rust
let mut sum = 0.0;                    // f64 로 추론
let cols: Vec<&str> = line.split(',').collect();   // 여기는 적었다
```

Rust는 타입을 추론한다. `0.0`을 보고 `f64`라고 안다.
그런데 `collect()` 줄에는 왜 적었을까?

```rust
let cols = line.split(',').collect();   // error[E0282]: type annotations needed
```

`collect()`는 **여러 타입으로 모을 수 있어서** 컴파일러가 고를 수 없다.

```rust
let a: Vec<&str>       = line.split(',').collect();
let b: String          = line.split(',').collect();
let c: HashSet<&str>   = line.split(',').collect();
```

이럴 때만 적어 준다. **"컴파일러가 모를 때만 적는다"**가 기준이다.

함수 시그니처는 예외다. 항상 적어야 한다.

```rust
fn magnitude(&self) -> f64 { ... }
//                     ^^^ 생략 불가
```

> rust-kr [3.2 데이터 타입](https://doc.rust-kr.org/ch03-02-data-types.html)

---

# 7. `Vec` — 늘어나는 배열

```rust
let mut samples = Vec::new();
samples.push(s);
samples.len()
samples[0]
```

`Vec<T>`는 C++의 `std::vector<T>`와 같다. 힙에 있고, 크기가 변한다.

```rust
let v: Vec<f64> = Vec::new();
let v = vec![1.0, 2.0, 3.0];      // 매크로로 한 번에
```

`Vec::new()`에 타입을 안 적었는데 되는 이유는,
아래에서 `samples.push(s)`를 보고 `Vec<Sample>`이라고 추론하기 때문이다.

배열 인덱스는 `usize` 타입이고, **범위를 벗어나면 패닉**한다.

```rust
let x = samples[999];
// thread 'main' panicked at: index out of bounds: the len is 5 but the index is 999
```

C처럼 조용히 엉뚱한 메모리를 읽지 않는다. Chapter 9에서 이걸 안전하게 다루는 법이 나온다.

> rust-kr [8.1 벡터](https://doc.rust-kr.org/ch08-01-vectors.html)

---

# 8. `for`와 반복

```rust
for line in raw.lines() { ... }
for s in &samples { ... }
for pair in samples.windows(2) { ... }
```

Rust의 `for`는 **항상 "무언가를 순회"**한다. C 스타일 `for(i=0; i<n; i++)`가 없다.

세 줄이 다 다른 것을 순회한다.

```text
raw.lines()          문자열을 줄 단위로
&samples             Vec 의 각 원소를 빌려서
samples.windows(2)   연속한 2개씩 겹치며
```

`windows(2)`가 마지막 검사를 우아하게 만든다.

```text
samples = [A, B, C, D, E]

windows(2) →  [A,B]  [B,C]  [C,D]  [D,E]
              연속한 쌍이 알아서 나온다
```

인덱스로 쓰면 이렇게 된다.

```rust
// 이렇게 쓸 수도 있지만
for i in 0..samples.len() - 1 {
    let dt = samples[i + 1].stamp - samples[i].stamp;
}
```

`samples`가 비어 있으면 `0 - 1`이 **underflow로 패닉**한다 (`usize`는 음수가 없다).
`windows(2)`는 빈 경우에 그냥 0번 돈다. **경계 조건을 안 틀리는 쪽을 고르는 게 낫다.**

`&samples`의 `&`는 Chapter 8의 주제다. 지금은 "빌려서 본다" 정도로 둔다.

> rust-kr [3.5 제어 흐름문](https://doc.rust-kr.org/ch03-05-control-flow.html),
> [13.2 반복자](https://doc.rust-kr.org/ch13-02-iterators.html)

---

# 9. `println!`과 포맷

```rust
println!("샘플 {}개", samples.len());
println!("평균 가속도 크기: {:.3} m/s^2", mean);
println!("최대: {:.3} at t={:.3}", worst.magnitude(), worst.stamp);
```

`!`가 붙으면 **매크로**다. 함수가 아니다.
인자 개수가 가변이고 컴파일 타임에 형식 문자열을 검사하기 때문에 매크로여야 한다.

```text
{}        기본 출력 (Display)
{:?}      디버그 출력 (Debug)   ← #[derive(Debug)] 필요
{:#?}     디버그 출력, 여러 줄
{:.3}     소수점 3자리
{:>8}     오른쪽 정렬 폭 8
{:.3e}    지수 표기
```

변수 이름을 직접 넣을 수도 있다.

```rust
let n = 5;
println!("샘플 {n}개");           // 짧다
println!("{:.3}", mean);
println!("{mean:.3}");            // 이것도 된다
```

형식이 안 맞으면 **컴파일 에러**다. C의 `printf`처럼 런타임에 깨지지 않는다.

```rust
println!("{:.3}", "hello");
// error[E0277]: the trait bound `str: std::fmt::Display` is not satisfied
```

> rust-kr [부록 B - 연산자와 기호](https://doc.rust-kr.org/appendix-02-operators.html)

---

# 10. `.parse()`와 `.unwrap()` — 지금은 넘어가는 부분

```rust
stamp: cols[0].parse().unwrap(),
```

`parse()`는 문자열을 숫자로 바꾼다. 그런데 **실패할 수 있다.**
`"abc".parse::<f64>()`는 어떻게 될까?

Rust는 이런 함수가 `Result`를 반환하게 한다.

```text
parse()  →  Result<f64, ParseFloatError>
             성공이면 값, 실패면 에러가 들어 있는 상자
```

`unwrap()`은 **"상자를 열되, 실패였으면 그냥 죽어라"**는 뜻이다.

```rust
let x: f64 = "abc".parse().unwrap();
// thread 'main' panicked at: called `Result::unwrap()` on an `Err` value: ParseFloatError
```

지금은 데이터가 확실하니 `unwrap()`으로 두었다.
**실제 프로그램에서는 이러면 안 된다.** Chapter 9 전체가 이 이야기다.

`parse()`가 `f64`인 줄 아는 이유는 `stamp: f64`라는 필드 타입에서 역추론하기 때문이다.
따로 못 정하는 자리에서는 이렇게 적는다.

```rust
let x = cols[0].parse::<f64>().unwrap();   // 터보피시 ::<>
```

> rust-kr [9.2 Result로 복구 가능한 에러 처리하기](https://doc.rust-kr.org/ch09-02-recoverable-errors-with-result.html)

---

# 11. `as` — 타입 변환은 자동이 아니다

```rust
let mean = sum / samples.len() as f64;
```

`samples.len()`은 `usize`, `sum`은 `f64`다. 그냥 나누면 에러다.

```rust
let mean = sum / samples.len();
// error[E0277]: cannot divide `f64` by `usize`
```

**Rust는 숫자 타입을 자동으로 섞지 않는다.** C에서 `int`와 `float`가 조용히 승격되던 것과 다르다.

```rust
let a: i32 = 5;
let b: i64 = a as i64;
let c: f64 = a as f64;
let d: u8  = 300 as u8;    // 44 — 잘린다! as 는 검사하지 않는다
```

`as`는 **자르기도 한다.** 안전하게 하려면 `try_into()`를 쓴다.

```rust
let d: u8 = 300i32.try_into().unwrap();   // 여기서 패닉 → 잘림을 알아챈다
```

정밀도 문제가 걱정되는 자리에서는 `as`를 습관적으로 쓰지 않는 편이 낫다.

> rust-kr [3.2 데이터 타입](https://doc.rust-kr.org/ch03-02-data-types.html)

---

# 12. `\` 줄 이음 — 사소하지만 헷갈리는 것

```rust
let raw = "\
0.000,0.01,-0.02,9.79
0.005,0.02,-0.01,9.81";
```

문자열 끝의 `\`는 **그 줄바꿈을 무시하라**는 뜻이다. 없으면 맨 앞에 빈 줄이 하나 들어간다.

여러 줄 문자열이 그냥 되는 것도 특징이다. C처럼 `"..." "..."`로 이어 붙이지 않아도 된다.

---

# 13. Mini Practice

프로그램을 조금씩 고쳐 보면서 컴파일러가 뭐라고 하는지 본다.

```bash
cd /tmp/imu_stats
```

```text
1) let mut samples → let samples 로 바꾸고 cargo run
   무슨 에러가 나고, 컴파일러가 뭐라고 제안하나?

2) #[derive(Debug)] 를 지우고
   println!("{:?}", samples[0]); 를 추가한 뒤 cargo run
   에러 메시지에 어떤 trait 이 없다고 나오나?

3) samples.len() as f64 에서 as f64 를 지우고 cargo run
   → E0277

4) raw 데이터 한 줄의 숫자를 abc 로 바꾸고 cargo run
   → 컴파일은 되는데 실행 중에 패닉한다. 왜 컴파일 타임에 못 잡을까?

5) expected_dt 를 0.05 로 바꾸면 "간격 이상" 이 사라지는지 확인

6) 마지막 루프를 windows(2) 대신 인덱스로 바꿔 보고,
   raw 를 빈 문자열 "" 로 만들면 어떻게 되는지 비교
```

4번이 이 chapter에서 가장 중요하다.
**`unwrap()`은 문제를 컴파일 타임에서 런타임으로 미루는 일**이다.
Chapter 9에서 이걸 제대로 처리한다.

---

# 14. 오늘의 핵심

```text
        지금까지 이 프로그램에 실제로 쓰인 것

  데이터        struct Sample { stamp: f64, ... }
                #[derive(Debug, Clone, Copy)]

  동작          impl Sample { fn magnitude(&self) -> f64 }
                세미콜론 없는 마지막 식이 반환값

  변수          let (불변)  /  let mut (가변)
                타입은 대개 추론. 애매할 때만 적는다

  컬렉션        Vec::new(), push, len, [i]
                범위를 벗어나면 패닉한다

  반복          for x in 무언가
                .lines()  &vec  .windows(2)

  출력          println!("{:.3}", x)   형식은 컴파일 타임에 검사

  변환          as f64      자동 승격이 없다
  실패          .unwrap()   지금은 넘어감 → Chapter 9
```

---

# 15. 반드시 구분할 것

```text
let  ≠  let mut
   기본이 불변이다. 다른 언어와 반대

struct  ≠  impl
   데이터와 동작을 따로 쓴다

println!  은 함수가 아니라 매크로
   ! 가 붙은 것은 전부 매크로

{}  ≠  {:?}
   Display vs Debug. Debug 는 derive 가 필요

&self  ≠  self
   빌리는가, 가져가는가 (Chapter 8)

as  ≠  안전한 변환
   as 는 조용히 자른다. try_into() 는 알려준다

컴파일 에러  ≠  런타임 패닉
   unwrap() 은 후자로 미루는 것

세미콜론 있음  ≠  없음
   없는 마지막 식이 반환값이 된다
```

---

# 16. Chapter 연결

```text
Chapter 1~6
구성과 빌드 — 숲

Chapter 7  ← 여기
돌아가는 프로그램 하나를 통째로 읽기

Chapter 8
같은 프로그램을 함수로 쪼갠다 → 소유권과 대여를 만난다

Chapter 9
raw 문자열을 진짜 파일에서 읽는다 → unwrap() 을 걷어낸다
```

프로그램은 계속 이어진다. 새 예제를 만들지 않는다.
