---
title: "Chapter 9. 진짜 파일을 읽으며 unwrap() 걷어내기"
importance: 10
---

> **Goal:** Chapter 8의 프로그램이 하드코딩된 문자열 대신 **실제 파일**을 읽게 바꾸면서,
> `Option`, `Result`, `?`, `match`, 커스텀 에러 타입이 왜 필요한지 순서대로 만난다.

Chapter 7·8의 데이터는 소스 코드 안에 박혀 있었다. 그래서 절대 실패하지 않았다.
파일을 읽는 순간 실패할 수 있는 곳이 한꺼번에 늘어난다.

```text
파일이 없다          경로를 잘못 줬다
줄이 깨졌다          열이 3개다, 숫자가 아니다
파일이 비었다        평균을 낼 게 없다
인자를 안 줬다       무슨 파일을 읽으라는 건지 모른다
```

`unwrap()`으로 뭉개면 전부 패닉이다. 이 chapter는 그걸 하나씩 걷어낸다.

> 참고: [Rust 프로그래밍 언어 (한국어판)](https://doc.rust-kr.org/title-page.html)

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo run -- <파일>` · `echo $?`

| 명령어                       | 하는 일                                     |
| :--------------------------- | :------------------------------------------ |
| `cargo run -- imu.csv`       | `--` 뒤는 **프로그램에 넘기는 인자**        |
| `echo $?`                    | 방금 프로그램의 종료 코드 (0=성공)          |
| `cargo run 2>/dev/null`      | 표준에러만 숨기기 (stdout/stderr 구분 확인) |
| `RUST_BACKTRACE=1 cargo run` | 패닉 시 호출 스택                           |
| `rustc --explain E0277`      | `?` 를 잘못 쓴 에러 설명                    |
| `cargo clippy`               | `unwrap()` 남용도 지적해준다                |

---

# 1. 먼저 순진하게 — 전부 `unwrap()`

```rust
use std::fs;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let raw = fs::read_to_string(path).unwrap();
    let n = raw.lines().count();
    println!("{n} lines");
}
```

테스트 파일을 만들고 돌려 본다.

```bash
printf '0.000,0.01,-0.02,9.79\n0.005,0.02,-0.01,9.81\n' > imu.csv
cargo run -- imu.csv
```

```text
2 lines
```

잘 된다. 그런데 조금만 어긋나면 이렇게 된다.

```bash
$ cargo run -- nope.csv
thread 'main' panicked at src/main.rs:6:40:
called `Result::unwrap()` on an `Err` value:
  Os { code: 2, kind: NotFound, message: "No such file or directory" }

$ cargo run
thread 'main' panicked at src/main.rs:5:40:
called `Option::unwrap()` on a `None` value
```

두 메시지가 다르다는 데 주목한다.

```text
Result::unwrap() on an `Err`     → 실패에 이유가 있다 (파일이 없음)
Option::unwrap() on a `None`     → 그냥 값이 없다 (이유랄 게 없다)
```

**Rust는 "없음"과 "실패"를 다른 타입으로 구분한다.** 이게 이 chapter의 뼈대다.

---

# 2. `Option<T>` — 있을 수도 없을 수도

```rust
std::env::args().nth(1)   //  Option<String>
```

`nth(1)`은 두 번째 인자를 준다. 없으면? 에러가 아니라 **그냥 없는 것**이다.

```rust
enum Option<T> {
    Some(T),
    None,
}
```

`enum`은 "이 중 하나"를 나타내는 타입이다. `Option`은 두 갈래뿐이다.

```text
Some(값)    값이 있다
None       값이 없다
```

**Rust에는 `null`이 없다.** 대신 값이 없을 수 있으면 타입에 `Option`이 적힌다.

```text
C++     Sample* p = find(...);
        p 가 nullptr 일 수 있다는 걸 타입만 봐서는 모른다
        검사를 잊으면 런타임에 터진다

Rust    fn find(...) -> Option<&Sample>
        Option 이므로 반드시 열어봐야 값을 쓸 수 있다
        검사를 잊으면 컴파일이 안 된다
```

> rust-kr [6.1 열거형 정의하기](https://doc.rust-kr.org/ch06-01-defining-an-enum.html)

---

# 3. `Result<T, E>` — 실패에 이유가 있다

```rust
fs::read_to_string(path)   //  Result<String, std::io::Error>
```

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

`Option`과 모양이 같은데, 실패 쪽에 **왜 실패했는지가 들어 있다.**

```text
Option<T>       Some(T)  |  None          이유 없음
Result<T, E>    Ok(T)    |  Err(E)        이유 있음
```

어느 쪽을 쓸지는 이 질문으로 갈린다.

```text
"실패했을 때 사용자에게 이유를 알려줘야 하나?"

  예   → Result    파일 읽기, 파싱, 네트워크
  아니오 → Option    빈 목록의 최댓값, 없을 수도 있는 설정값
```

> rust-kr [9.2 Result로 복구 가능한 에러 처리하기](https://doc.rust-kr.org/ch09-02-recoverable-errors-with-result.html)

---

# 4. 상자를 여는 방법들

`Option`이든 `Result`든 값을 쓰려면 열어야 한다. 방법이 여러 가지다.

```rust
// ① unwrap — 실패면 패닉. 지금까지 쓰던 것
let p = args.nth(1).unwrap();

// ② expect — 패닉하되 메시지를 남긴다
let p = args.nth(1).expect("파일 경로를 인자로 주세요");

// ③ unwrap_or — 실패면 기본값
let level = env::var("LOG").unwrap_or("info".to_string());

// ④ match — 갈래마다 다르게 처리
match args.nth(1) {
    Some(p) => p,
    None => { eprintln!("사용법: imu_stats <파일>"); process::exit(2); }
}

// ⑤ if let — 한쪽만 관심 있을 때
if let Some(w) = worst_sample(&samples) {
    println!("최대: {:.3}", w.magnitude());
}

// ⑥ ? — 실패면 그대로 위로 넘긴다 (§6)
let raw = fs::read_to_string(path)?;
```

언제 무엇을 쓰나.

```text
①  unwrap        프로토타입, 테스트, "여기서 실패하면 버그다" 인 곳
②  expect        같지만 왜 확신하는지 메시지로 남긴다  ← unwrap 보다 항상 낫다
③  unwrap_or     기본값이 자연스러운 경우
④  match         갈래마다 정말 다르게 처리할 때
⑤  if let        한쪽만 필요하고 나머지는 무시할 때
⑥  ?             호출한 쪽이 처리하게 넘길 때        ← 라이브러리 코드의 기본
```

**`unwrap()` 대신 `expect("...")`를 쓰는 습관**만 들여도 디버깅이 훨씬 쉬워진다.
패닉했을 때 파일·줄 번호와 함께 내가 쓴 문장이 나온다.

> rust-kr [6.2 match](https://doc.rust-kr.org/ch06-02-match.html),
> [6.3 if let](https://doc.rust-kr.org/ch06-03-if-let.html)

---

# 5. `match`는 빠뜨릴 수 없다

`match`의 핵심은 **모든 갈래를 다루지 않으면 컴파일이 안 된다**는 것이다.

```rust
match mean_magnitude(&samples) {
    Some(m) => println!("평균: {m:.3}"),
    // None 을 빼먹으면?
}
```

```text
error[E0004]: non-exhaustive patterns: `None` not covered
```

C의 `switch`에서 `case`를 빠뜨려도 조용히 통과하던 것과 다르다.
나중에 `enum`에 갈래를 하나 추가하면, **그걸 처리하지 않은 모든 `match`가 컴파일 에러**가 된다.
어디를 고쳐야 하는지 컴파일러가 전부 찾아준다.

관심 없는 나머지는 `_`로 묶는다.

```rust
match n {
    0 => println!("없음"),
    1 => println!("하나"),
    _ => println!("여러 개"),
}
```

> rust-kr [6.2 match 제어 흐름 구조](https://doc.rust-kr.org/ch06-02-match.html)

---

# 6. `?` — 실패를 위로 넘기는 연산자

가장 많이 쓰게 될 문법이다.

```rust
let raw = fs::read_to_string(path)?;
```

이 한 글자가 하는 일은 이렇다.

```rust
// ? 를 풀어 쓰면
let raw = match fs::read_to_string(path) {
    Ok(v) => v,
    Err(e) => return Err(e.into()),
};
```

```text
성공이면  →  값을 꺼내서 계속 진행
실패면    →  즉시 return 하고 에러를 호출한 쪽에 넘긴다
```

**성공 경로만 코드에 남는다**는 게 핵심이다. 에러 처리 코드가 본문을 어지럽히지 않는다.

```rust
fn load(path: &str) -> Result<Vec<Sample>, LogError> {
    let raw = fs::read_to_string(path)?;          // 실패하면 여기서 끝
    let mut samples = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        samples.push(parse_line(line, i + 1)?);   // 여기서도
    }
    Ok(samples)
}
```

단, `?`는 **`Result`나 `Option`을 반환하는 함수 안에서만** 쓸 수 있다.

```rust
fn main() {
    let raw = fs::read_to_string("imu.csv")?;   // ✗
}
```

```text
error[E0277]: the `?` operator can only be used in a function that
              returns `Result` or `Option`
  |
2 | fn main() {
  | --------- this function should return `Result` or `Option` to accept `?`
  |
help: consider adding return type
  |
2 ~ fn main() -> Result<(), Box<dyn std::error::Error>> {
```

여기서도 컴파일러가 고치는 법을 알려준다.

> rust-kr [9.2 Result](https://doc.rust-kr.org/ch09-02-recoverable-errors-with-result.html)

---

# 7. 내 에러 타입 만들기

`?`로 넘기려면 에러 타입이 하나로 정해져야 한다.
그런데 우리 프로그램에는 실패 종류가 여러 가지다.

```text
파일을 못 읽음       std::io::Error
줄이 깨짐            우리가 정의해야 함
파일이 빔            우리가 정의해야 함
```

`enum`으로 묶는다.

```rust
#[derive(Debug)]
enum LogError {
    Io(std::io::Error),
    BadLine { line_no: usize, reason: String },
    Empty,
}
```

`enum`의 각 갈래가 **데이터를 가질 수 있다**는 것이 Rust `enum`의 강점이다.

```text
Io(std::io::Error)                    값 하나를 품는다
BadLine { line_no, reason }           구조체처럼 이름 붙은 필드
Empty                                 아무것도 없음
```

C의 `enum`은 정수 상수 목록일 뿐이지만, Rust의 `enum`은 **갈래마다 다른 데이터**를 담는다.

사람이 읽을 메시지는 `Display`로 정한다.

```rust
use std::fmt;

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LogError::Io(e) => write!(f, "파일을 읽을 수 없습니다: {e}"),
            LogError::BadLine { line_no, reason } =>
                write!(f, "{line_no}번째 줄이 잘못되었습니다: {reason}"),
            LogError::Empty => write!(f, "샘플이 하나도 없습니다"),
        }
    }
}
```

```text
Debug   개발자용.  {:?}   #[derive(Debug)] 로 충분
Display 사용자용. {}     직접 써야 한다
```

> rust-kr [6.1 열거형](https://doc.rust-kr.org/ch06-01-defining-an-enum.html),
> [10.2 트레이트](https://doc.rust-kr.org/ch10-02-traits.html)

---

# 8. `From` — `?`가 타입을 알아서 바꾸게

`fs::read_to_string`은 `io::Error`를 주는데 우리 함수는 `LogError`를 반환한다.
그냥 `?`를 쓰면 타입이 안 맞는다. 변환 규칙을 알려주면 된다.

```rust
impl From<std::io::Error> for LogError {
    fn from(e: std::io::Error) -> Self {
        LogError::Io(e)
    }
}
```

이것만 있으면 `?`가 알아서 변환한다. (§6의 `e.into()`가 이걸 부른다.)

```rust
let raw = fs::read_to_string(path)?;   // io::Error → LogError 자동 변환
```

`From`이 없을 때는 `map_err`로 직접 바꾼다.

```rust
v[i] = c.trim().parse().map_err(|_| LogError::BadLine {
    line_no,
    reason: format!("숫자가 아닙니다: {:?}", c),
})?;
```

```text
map_err(|e| ...)   Err 쪽만 다른 값으로 바꾼다. Ok 면 그대로 통과
|_| ...            클로저. _ 는 "받은 에러는 안 쓴다"는 뜻
```

여기서는 원래 에러(`ParseFloatError`)보다 **몇 번째 줄인지**가 더 유용하므로 버리고 새로 만들었다.

> rust-kr [13.1 클로저](https://doc.rust-kr.org/ch13-01-closures.html)

---

# 9. 완성된 프로그램

```rust
use std::fmt;
use std::fs;
use std::process;

#[derive(Debug, Clone, Copy)]
struct Sample { stamp: f64, ax: f64, ay: f64, az: f64 }

impl Sample {
    fn magnitude(&self) -> f64 {
        (self.ax * self.ax + self.ay * self.ay + self.az * self.az).sqrt()
    }
}

#[derive(Debug)]
enum LogError {
    Io(std::io::Error),
    BadLine { line_no: usize, reason: String },
    Empty,
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LogError::Io(e) => write!(f, "파일을 읽을 수 없습니다: {e}"),
            LogError::BadLine { line_no, reason } =>
                write!(f, "{line_no}번째 줄이 잘못되었습니다: {reason}"),
            LogError::Empty => write!(f, "샘플이 하나도 없습니다"),
        }
    }
}

impl From<std::io::Error> for LogError {
    fn from(e: std::io::Error) -> Self { LogError::Io(e) }
}

fn parse_line(line: &str, line_no: usize) -> Result<Sample, LogError> {
    let cols: Vec<&str> = line.split(',').collect();
    if cols.len() != 4 {
        return Err(LogError::BadLine {
            line_no,
            reason: format!("열이 4개여야 하는데 {}개입니다", cols.len()),
        });
    }
    let mut v = [0.0f64; 4];
    for (i, c) in cols.iter().enumerate() {
        v[i] = c.trim().parse().map_err(|_| LogError::BadLine {
            line_no,
            reason: format!("숫자가 아닙니다: {:?}", c),
        })?;
    }
    Ok(Sample { stamp: v[0], ax: v[1], ay: v[2], az: v[3] })
}

fn load(path: &str) -> Result<Vec<Sample>, LogError> {
    let raw = fs::read_to_string(path)?;
    let mut samples = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        samples.push(parse_line(line, i + 1)?);
    }
    if samples.is_empty() { return Err(LogError::Empty); }
    Ok(samples)
}

fn mean_magnitude(samples: &[Sample]) -> Option<f64> {
    if samples.is_empty() { return None; }
    let sum: f64 = samples.iter().map(|s| s.magnitude()).sum();
    Some(sum / samples.len() as f64)
}

fn worst_sample(samples: &[Sample]) -> Option<&Sample> {
    samples.iter().max_by(|a, b| a.magnitude().total_cmp(&b.magnitude()))
}

fn find_gaps(samples: &[Sample], expected_dt: f64) -> Vec<(f64, f64)> {
    samples.windows(2)
        .map(|p| (p[0].stamp, p[1].stamp - p[0].stamp))
        .filter(|(_, dt)| *dt > expected_dt * 1.5)
        .collect()
}

fn run(path: &str) -> Result<(), LogError> {
    let samples = load(path)?;
    println!("샘플 {}개", samples.len());

    match mean_magnitude(&samples) {
        Some(m) => println!("평균: {m:.3}"),
        None => println!("평균: 계산할 수 없음"),
    }

    if let Some(w) = worst_sample(&samples) {
        println!("최대: {:.3} at t={:.3}", w.magnitude(), w.stamp);
    }

    let gaps = find_gaps(&samples, 0.005);
    if gaps.is_empty() {
        println!("간격 이상 없음");
    } else {
        for (t, dt) in gaps { println!("간격 이상: t={t:.3} 에서 {dt:.3}s"); }
    }
    Ok(())
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("사용법: imu_stats <파일>");
            process::exit(2);
        }
    };

    if let Err(e) = run(&path) {
        eprintln!("에러: {e}");
        process::exit(1);
    }
}
```

---

# 10. 실제로 돌려 보기

```bash
printf '0.000,0.01,-0.02,9.79\n0.005,0.02,-0.01,9.81\n0.010,0.15,0.30,10.90\n0.015,0.01,0.00,9.80\n0.045,-0.02,0.01,9.78\n' > imu.csv
printf '0.000,0.01,-0.02,9.79\nabc,0.02,-0.01,9.81\n' > bad.csv
printf '0.000,0.01,-0.02\n' > short.csv
: > empty.csv
```

```text
$ cargo run -q -- imu.csv
샘플 5개
평균: 10.017
최대: 10.905 at t=0.010
간격 이상: t=0.015 에서 0.030s
$ echo $?
0

$ cargo run -q -- bad.csv
에러: 2번째 줄이 잘못되었습니다: 숫자가 아닙니다: "abc"
$ echo $?
1

$ cargo run -q -- short.csv
에러: 1번째 줄이 잘못되었습니다: 열이 4개여야 하는데 3개입니다

$ cargo run -q -- empty.csv
에러: 샘플이 하나도 없습니다

$ cargo run -q -- nope.csv
에러: 파일을 읽을 수 없습니다: No such file or directory (os error 2)

$ cargo run -q
사용법: imu_stats <파일>
$ echo $?
2
```

**패닉이 하나도 없다.** 전부 사람이 읽을 수 있는 메시지와 종료 코드로 끝난다.
Chapter 7의 `unwrap()` 버전과 비교해 보면 차이가 분명하다.

```text
Chapter 7        thread 'main' panicked at src/main.rs:23:37:
                 called `Result::unwrap()` on an `Err` value: ParseFloatError

Chapter 9        에러: 2번째 줄이 잘못되었습니다: 숫자가 아닙니다: "abc"
```

---

# 11. `eprintln!`과 종료 코드

에러는 **표준출력이 아니라 표준에러로** 낸다.

```rust
println!("샘플 5개");     // stdout — 정상 결과
eprintln!("에러: {e}");   // stderr — 에러 메시지
```

이렇게 해야 파이프로 넘길 때 결과만 깨끗하게 걸러진다.

```bash
cargo run -q -- imu.csv > result.txt      # 에러는 화면에 그대로 보인다
cargo run -q -- bad.csv 2>/dev/null       # 에러만 숨긴다
```

종료 코드도 의미를 나눠 둔다.

```text
0    성공
1    실행 중 실패 (파일이 없다, 데이터가 깨졌다)
2    사용법 오류 (인자를 안 줬다)
```

스크립트나 systemd가 이 값으로 판단한다.
Edge Computing Chapter 19에서 `docker ps -a`의 `Exited (1)`을 보던 그 숫자다.

> rust-kr [12.6 표준 출력 대신 표준 에러로](https://doc.rust-kr.org/ch12-06-writing-to-stderr-instead-of-stdout.html)

---

# 12. 새로 나온 반복자 표현들

`find_gaps`가 Chapter 8보다 짧아졌다.

```rust
// Chapter 8
fn find_gaps(samples: &[Sample], expected_dt: f64) -> Vec<(f64, f64)> {
    let mut gaps = Vec::new();
    for pair in samples.windows(2) {
        let dt = pair[1].stamp - pair[0].stamp;
        if dt > expected_dt * 1.5 { gaps.push((pair[0].stamp, dt)); }
    }
    gaps
}

// Chapter 9
fn find_gaps(samples: &[Sample], expected_dt: f64) -> Vec<(f64, f64)> {
    samples.windows(2)
        .map(|p| (p[0].stamp, p[1].stamp - p[0].stamp))
        .filter(|(_, dt)| *dt > expected_dt * 1.5)
        .collect()
}
```

```text
.map(|x| ...)      각 원소를 다른 것으로 바꾼다
.filter(|x| ...)   조건에 맞는 것만 남긴다
.collect()         결과를 Vec 등으로 모은다   ← Chapter 7 §6 의 그것
.sum()             전부 더한다
.max_by(...)       최댓값. 없으면 None       ← Option 을 준다
```

`|p| ...`는 **클로저**, 이름 없는 함수다.

```text
|p| p[0].stamp        인자 하나, 값 하나 반환
|a, b| a + b          인자 둘
|_| ...               인자를 안 쓴다
```

`worst_sample`도 루프가 통째로 사라졌다.

```rust
samples.iter().max_by(|a, b| a.magnitude().total_cmp(&b.magnitude()))
```

`max_by`가 빈 slice에 대해 자동으로 `None`을 준다.
**Chapter 8에서 `&samples[0]`이 빈 입력에 패닉하던 문제가 저절로 사라졌다.**

`total_cmp`를 쓴 이유는 `f64`가 `NaN` 때문에 완전한 순서가 아니라서다.
`partial_cmp`를 쓰면 `Option`이 또 나와서 번거로워진다.

> rust-kr [13.1 클로저](https://doc.rust-kr.org/ch13-01-closures.html),
> [13.2 반복자](https://doc.rust-kr.org/ch13-02-iterators.html)

---

# 13. Mini Practice

```bash
cd /tmp/imu_stats
```

```text
1) main 안에서 fs::read_to_string("imu.csv")? 를 직접 써 보기
   → error[E0277]: the `?` operator can only be used in a function
                   that returns `Result` or `Option`
   컴파일러가 제안하는 수정(fn main() -> Result<(), Box<dyn Error>>)을
   적용하면 되는지 확인

2) run() 의 match 에서 None 갈래를 지우고 cargo check
   → error[E0004]: non-exhaustive patterns: `None` not covered

3) impl From<std::io::Error> for LogError 를 통째로 지우고 cargo check
   → fs::read_to_string(path)? 가 왜 깨지나?

4) parse_line 의 map_err(...) 를 지우고 그냥 ? 만 남기면?
   → 어떤 타입이 안 맞는다고 하나?

5) 열 개수 검사(cols.len() != 4)를 지우고 short.csv 로 실행
   → 패닉이 안 난다. 대신 이렇게 나온다:
         샘플 1개
         평균: 0.022
      az 가 조용히 0.0 으로 채워진 결과다. 왜 그럴까?

   같은 파일을 Chapter 7 방식(cols[3] 로 직접 인덱싱)으로 읽으면:
         panicked at: index out of bounds: the len is 3 but the index is 3

6) 종료 코드를 확인
       cargo run -q -- imu.csv;  echo $?     → 0
       cargo run -q -- bad.csv;  echo $?     → 1
       cargo run -q;             echo $?     → 2

7) stdout 과 stderr 를 분리해 보기
       cargo run -q -- bad.csv > out.txt
       cat out.txt          # 비어 있어야 정상
```

3번과 5번이 핵심이다.

3번은 `From`이 `?`를 뒤에서 떠받치고 있다는 것을 보여준다.
지우면 `error[E0277]: \`?\` couldn't convert the error to \`LogError\``가 뜬다.

5번은 더 무서운 것을 보여준다.

```text
Chapter 7 방식 (cols[3])          →  패닉. 시끄럽지만 즉시 알아챈다
검사만 지운 Chapter 9 방식         →  az = 0.0 으로 조용히 채워짐
                                     결과가 그럴듯해서 못 알아챈다
```

`for (i, c) in cols.iter().enumerate()`는 있는 열만큼만 돈다.
`v`는 `[0.0f64; 4]`로 초기화되어 있으므로 없는 열은 그냥 0.0으로 남는다.
**컴파일도 되고 실행도 되는데 답이 틀린다.**

로봇 로그라면 이건 "가속도 z축이 0"이라는 물리적으로 말이 안 되는 데이터가
아무 경고 없이 파이프라인을 타고 흘러간다는 뜻이다.
`Result`가 자동으로 막아주지 않는 종류의 버그이고,
**`cols.len() != 4` 같은 검사를 사람이 직접 써야 하는 이유**다.

---

# 14. 오늘의 핵심

```text
                     실패를 표현하는 두 타입

   Option<T>              Result<T, E>
   ─────────────          ──────────────────────
   Some(T) | None         Ok(T) | Err(E)
   "없다"                 "실패했다, 이유는 E"

                     여는 방법

   unwrap()      실패면 패닉        프로토타입
   expect("..")  패닉 + 내 메시지    unwrap 보다 항상 낫다
   unwrap_or(x)  실패면 기본값
   match         갈래마다 처리       빠뜨리면 컴파일 에러
   if let        한쪽만 관심
   ?             위로 넘긴다        ← 라이브러리의 기본

                     ? 가 동작하려면

   함수가 Result/Option 을 반환해야 하고
   From<원래에러> for 내에러  가 있어야 자동 변환된다

                     경계에서

   run() -> Result       실패를 위로 모은다
   main()                여기서 한 번만 메시지 + 종료 코드로 바꾼다
```

---

# 15. 반드시 구분할 것

```text
Option  ≠  Result
   "없음" vs "이유 있는 실패"

null  ≠  Option
   Rust 에 null 은 없다. 없을 수 있으면 타입에 적힌다

unwrap()  ≠  expect("...")
   후자는 왜 확신했는지가 패닉 메시지에 남는다

Debug  ≠  Display
   {:?} 개발자용(derive 가능) vs {} 사용자용(직접 구현)

?  는 아무 데서나 못 쓴다
   Result/Option 을 반환하는 함수 안에서만

map_err  ≠  unwrap
   에러를 "바꾸는" 것이지 "여는" 것이 아니다

match 는 빠뜨릴 수 없다
   enum 에 갈래를 추가하면 고쳐야 할 곳을 전부 알려준다

println!  ≠  eprintln!
   stdout(결과) vs stderr(에러)

samples[0]  은 여전히 패닉한다
   Option 을 쓰려면 .first() / .get(0)

타입이 안전함  ≠  값이 올바름
   열이 모자라도 0.0 으로 채워지면 컴파일러는 아무 말도 안 한다
   입력 검증은 여전히 사람 몫이다
```

---

# 16. Chapter 연결

```text
Chapter 7
프로그램 한 덩어리 — unwrap() 으로 뭉갬

Chapter 8
함수로 쪼갬 — 소유권, 대여

Chapter 9  ← 여기
진짜 파일을 읽음 — Option, Result, ?, match, 커스텀 에러

(다음 후보)
trait 과 제네릭   LiDAR 를 추가하면서 공통 동작을 뽑아내기
테스트           cargo test 로 parse_line 의 실패 경로를 검증
반복자           map / filter / collect 를 더 깊이

Edge Computing Chapter 19
종료 코드와 자동 재시작 — 여기서 만든 exit(1) 이 쓰이는 곳
```
