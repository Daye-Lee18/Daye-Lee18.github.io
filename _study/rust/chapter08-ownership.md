---
title: "Chapter 8. 함수로 쪼개다가 소유권을 만나기"
importance: 9
---

> **Goal:** Chapter 7의 프로그램을 함수로 나누면서 `E0382`, `E0499`, `E0502`를 직접 만나고,
> `&`, `&mut`, `&[T]`를 언제 쓰는지 몸으로 익힌다.

Chapter 7의 `main`은 전부 한 함수에 들어 있었다. 그래서 소유권 문제가 안 생겼다.
**함수로 쪼개는 순간 문제가 시작된다.** 그리고 그게 Rust를 배우는 가장 자연스러운 지점이다.

새 예제를 만들지 않는다. **같은 프로그램을 리팩터링한다.**

---

# 0. 자주 쓰는 명령어

> 이 chapter에서 가장 많이 치는 것: `cargo check` · `rustc --explain E0382`

| 명령어                  | 하는 일                              |
| :---------------------- | :----------------------------------- |
| `cargo check`           | 소유권 에러는 여기서 다 잡힌다       |
| `rustc --explain E0382` | **move 에러 설명**                   |
| `rustc --explain E0499` | 가변 대여 두 번                      |
| `rustc --explain E0502` | 불변·가변 대여 충돌                  |
| `rustc --explain E0596` | `mut`이 아닌데 가변으로 빌리려 함    |
| `cargo clippy`          | `&Vec<T>` 대신 `&[T]` 같은 것을 제안 |

---

# 1. 목표 — 이렇게 쪼개고 싶다

Chapter 7의 `main`을 기능별로 나누면 이런 모양이 된다.

```rust
fn parse_all(raw: &str) -> Vec<Sample> { ... }
fn mean_magnitude(samples: &[Sample]) -> f64 { ... }
fn worst_sample(samples: &[Sample]) -> &Sample { ... }
fn find_gaps(samples: &[Sample], expected_dt: f64) -> Vec<(f64, f64)> { ... }
fn remove_gravity(samples: &mut Vec<Sample>, g: f64) { ... }

fn main() {
    let mut samples = parse_all(raw);
    println!("평균: {:.3}", mean_magnitude(&samples));
    ...
    remove_gravity(&mut samples, 9.81);
}
```

시그니처에 `&`, `&[T]`, `&mut`가 섞여 있다.
**왜 각각 저것이어야 하는지**가 이 chapter의 내용이다.

먼저 순진하게 써 보고, 컴파일러에게 혼나면서 고쳐 간다.

---

# 2. 첫 시도 — 그냥 값으로 넘기면

```rust
fn mean_magnitude(samples: Vec<Sample>) -> f64 { ... }   // & 없이
fn count(samples: Vec<Sample>) -> usize { ... }

fn main() {
    let samples = parse_all(raw);
    let m = mean_magnitude(samples);
    let c = count(samples);          // ← 여기서 막힌다
}
```

```text
error[E0382]: use of moved value: `samples`
  |
6 |     let samples = vec![...];
  |         ------- move occurs because `samples` has type `Vec<Sample>`,
  |                 which does not implement the `Copy` trait
7 |     let m = mean_magnitude(samples);
  |                            ------- value moved here
8 |     let c = count(samples);
  |                   ^^^^^^^ value used here after move
  |
note: consider changing this parameter type in function `mean_magnitude`
      to borrow instead if owning the value isn't necessary
```

Rust를 처음 하면 여기서 한 번 멈춘다. **"넘겼을 뿐인데 왜 없어졌지?"**

> rust-kr [4.1 소유권이 뭔가요?](https://doc.rust-kr.org/ch04-01-what-is-ownership.html)

---

# 3. 왜 사라지는가 — 소유권

Rust의 규칙은 세 줄이다.

```text
① 모든 값에는 소유자(owner)가 정확히 하나 있다
② 소유자가 스코프를 벗어나면 값은 버려진다(drop)
③ 소유권은 넘어간다(move). 넘기면 원래 자리에서는 못 쓴다
```

`mean_magnitude(samples)`는 **소유권을 넘긴 것**이다.

```text
호출 전                 호출 중                    호출 후
─────────────          ──────────────────         ──────────────
main:                   mean_magnitude:            main:
  samples ──▶ [데이터]     samples ──▶ [데이터]        samples ──▶ ✗
                                                     (더 이상 소유자가 아니다)
                        함수가 끝나면 여기서 drop
```

왜 이렇게 만들었을까. **메모리를 언제 해제할지 컴파일 타임에 정하기 위해서**다.

```text
C/C++     free() 를 사람이 부른다   → 안 부르면 누수, 두 번 부르면 double free
GC 언어    런타임이 알아서 치운다     → 언제 멈출지 예측이 어렵다
Rust      소유자가 스코프를 벗어날 때 → 컴파일 타임에 결정. GC 없음
```

로봇 제어 루프처럼 **멈추면 안 되는 코드**에서 GC가 없다는 것이 Rust를 쓰는 큰 이유다.
(Edge Computing Chapter 15의 실시간 이야기와 이어진다.)

---

# 4. `Copy`가 붙은 것은 왜 안 사라지나

그런데 이건 잘 된다.

```rust
let a = 5;
let b = a;
println!("{a} {b}");     // 둘 다 잘 나온다
```

`i32`처럼 **작고 스택에만 있는 값**은 `Copy` trait을 구현하고 있다.
`Copy`가 있으면 move 대신 **복사**가 일어난다.

```text
Copy 인 것          i32, f64, bool, char, &T,
                   그리고 Copy 인 것들로만 이루어진 struct/tuple

Copy 가 아닌 것     Vec<T>, String, Box<T>, 파일 핸들 …
                   힙을 쓰거나 해제가 필요한 것
```

Chapter 7에서 `Sample`에 `#[derive(Clone, Copy)]`를 붙여 뒀던 것이 여기서 효과를 낸다.

```rust
#[derive(Debug, Clone, Copy)]
struct Sample { stamp: f64, ax: f64, ay: f64, az: f64 }
```

`Sample`은 `f64` 네 개뿐이라 32바이트다. 복사가 싸므로 `Copy`가 적절하다.
그래서 `Sample` 하나를 넘기는 것은 문제가 안 된다.

**하지만 `Vec<Sample>`은 여전히 `Copy`가 아니다.** 힙에 있기 때문이다.

```text
Sample          32 바이트, 스택        → Copy 로 둘 만하다
Vec<Sample>     힙 할당 + 길이 + 용량   → Copy 불가. move 된다
```

에러 메시지가 정확히 그 말을 하고 있었다.

```text
`Vec<Sample>`, which does not implement the `Copy` trait
```

> rust-kr [4.1 소유권](https://doc.rust-kr.org/ch04-01-what-is-ownership.html)

---

# 5. 고치는 법 세 가지, 그리고 왜 `&`인가

```rust
// ① clone — 복사본을 만든다
let m = mean_magnitude(samples.clone());
let c = count(samples);
```

```text
된다. 하지만 데이터를 통째로 복사한다.
샘플이 100만 개면 100만 개를 복사한다. 낭비다.
```

```rust
// ② 돌려받기 — 소유권을 넘겼다가 되받는다
fn mean_magnitude(samples: Vec<Sample>) -> (f64, Vec<Sample>) { ... }
let (m, samples) = mean_magnitude(samples);
```

```text
된다. 하지만 시그니처가 흉해진다. 실제로 이렇게 쓰지 않는다.
```

```rust
// ③ 빌리기(borrow) — 소유권은 그대로 두고 보기만 한다
fn mean_magnitude(samples: &Vec<Sample>) -> f64 { ... }
let m = mean_magnitude(&samples);
let c = count(&samples);        // 몇 번이든 된다
```

**③이 정답이다.** `&`는 "잠깐 빌려서 읽는다"는 뜻이다.

```text
소유권을 넘긴다(move)      값을 통째로 준다. 원래 자리는 비어버린다
빌린다(borrow, &)          주소만 준다. 소유자는 그대로
```

C++의 `const&`와 비슷하지만, **Rust는 빌린 것이 소유자보다 오래 살지 못하도록 컴파일러가 강제한다.**
C++에서 dangling reference가 되는 코드가 Rust에서는 컴파일이 안 된다.

> rust-kr [4.2 참조와 대여](https://doc.rust-kr.org/ch04-02-references-and-borrowing.html)

---

# 6. `&Vec<T>` 대신 `&[T]` — clippy가 시키는 것

③으로 고치고 `cargo clippy`를 돌리면 이런 말을 듣는다.

```text
warning: writing `&Vec` instead of `&[_]` involves a new object where a slice will do
help: change this to: `&[Sample]`
```

`&[T]`는 **슬라이스**다. "연속된 T들의 어딘가"를 가리킨다.

```text
&Vec<Sample>    반드시 Vec 이어야 한다
&[Sample]       Vec 이든 배열이든 그 일부든 다 받는다
```

```rust
fn mean_magnitude(samples: &[Sample]) -> f64 { ... }

let v: Vec<Sample> = ...;
let a: [Sample; 3] = ...;

mean_magnitude(&v);          // Vec 에서
mean_magnitude(&a);          // 배열에서
mean_magnitude(&v[1..3]);    // 일부만
```

**읽기만 하는 함수는 `&[T]`로 받는 것이 기본**이다. 더 많은 것을 받으면서 잃는 게 없다.
문자열도 같은 관계다.

```text
&String   →  &str        같은 이유로 &str 을 쓴다
&Vec<T>   →  &[T]
```

Chapter 7의 `parse_all(raw: &str)`이 `&String`이 아니었던 이유다.

> rust-kr [4.3 슬라이스](https://doc.rust-kr.org/ch04-03-slices.html)

---

# 7. 고쳐야 할 때는 `&mut`

`remove_gravity`는 데이터를 **바꿔야** 한다. 읽기만으로는 안 된다.

```rust
fn remove_gravity(samples: &mut Vec<Sample>, g: f64) {
    for s in samples.iter_mut() {
        s.az -= g;
    }
}

remove_gravity(&mut samples, 9.81);
```

세 군데에 `mut`이 필요하다.

```text
let mut samples = ...;                       ① 원본이 가변이어야 한다
fn remove_gravity(samples: &mut Vec<Sample>) ② 가변으로 빌리겠다고 선언
remove_gravity(&mut samples, 9.81);          ③ 호출할 때도 명시
```

**호출하는 쪽에도 `&mut`을 쓰는 것**이 특징이다. C++의 `foo(x)`가 `x`를 바꿀 수 있는지
호출부만 봐서는 모르는 것과 다르다. Rust는 **읽는 사람이 바로 안다.**

`iter_mut()`도 마찬가지다. 그냥 `for s in samples`면 읽기만 된다.

```rust
for s in samples.iter()     { /* &Sample     읽기만 */ }
for s in samples.iter_mut() { /* &mut Sample 고칠 수 있다 */ }
for s in samples.into_iter(){ /* Sample      가져가버린다 */ }
```

---

# 8. 대여 규칙 — 왜 이렇게 까다로운가

Rust는 어느 시점에든 다음 중 **하나만** 허용한다.

```text
· 불변 참조(&T) 를 여러 개        (읽기만 하니 여럿이어도 안전)
· 가변 참조(&mut T) 를 딱 하나    (쓰는 건 하나만)

  둘을 섞을 수 없다.
```

어기면 이렇게 된다.

```rust
let mut v = vec![1.0, 2.0];
let a = &mut v;
let b = &mut v;
a.push(3.0);
```

```text
error[E0499]: cannot borrow `v` as mutable more than once at a time
  |
3 |     let a = &mut v;
  |             ------ first mutable borrow occurs here
4 |     let b = &mut v;
  |             ^^^^^^ second mutable borrow occurs here
```

섞어도 안 된다.

```rust
let mut data = vec![1.0, 2.0, 3.0];
let first = &data[0];      // 불변 대여
data.push(4.0);            // 가변 대여
println!("{first}");
```

```text
error[E0502]: cannot borrow `data` as mutable because it is also borrowed as immutable
  |
3 |     let first = &data[0];
  |                  ---- immutable borrow occurs here
4 |     data.push(4.0);
  |     ^^^^^^^^^^^^^^ mutable borrow occurs here
5 |     println!("{first}");
  |                ----- immutable borrow later used here
```

**이건 진짜 버그다.** `push`가 용량을 넘기면 `Vec`은 힙에 새 공간을 잡고 옮긴다.
그러면 `first`가 가리키던 주소는 해제된 메모리가 된다.

```text
C++ 에서 같은 코드
  std::vector<double> v = {1,2,3};
  double* first = &v[0];
  v.push_back(4);          // 재할당 → first 가 dangling
  std::cout << *first;     // undefined behavior. 운이 나쁘면 그냥 돈다

Rust
  컴파일이 안 된다
```

까다로운 게 아니라, **C++에서 런타임에 터지던 것을 컴파일 타임에 잡는 것**이다.

> rust-kr [4.2 참조와 대여](https://doc.rust-kr.org/ch04-02-references-and-borrowing.html)

---

# 9. 참조를 반환하기 — `worst_sample`

```rust
fn worst_sample(samples: &[Sample]) -> &Sample {
    let mut worst = &samples[0];
    for s in samples {
        if s.magnitude() > worst.magnitude() { worst = s; }
    }
    worst
}
```

참조를 **반환**하는데도 컴파일이 된다. 왜 안전할까?

반환된 `&Sample`은 인자로 받은 `samples` 안의 한 원소를 가리킨다.
컴파일러는 "**반환값은 입력이 살아 있는 동안만 유효하다**"고 기록해 둔다.

```rust
let w = worst_sample(&samples);
drop(samples);          // samples 를 없애면
println!("{:?}", w);    // ← error: borrow of moved value
```

이 규칙을 **라이프타임**이라고 부른다.
입력 참조가 하나뿐이면 컴파일러가 알아서 연결해 주기 때문에 여기서는 안 적어도 된다.
입력 참조가 둘 이상이면 직접 적어야 한다.

```rust
fn longer<'a>(a: &'a [Sample], b: &'a [Sample]) -> &'a [Sample] {
    if a.len() > b.len() { a } else { b }
}
```

`'a`는 "이 세 참조의 수명이 같이 간다"는 표시다.
**지금은 이 정도만 알면 된다.** 자세한 것은 필요해질 때 본다.

> rust-kr [10.3 라이프타임](https://doc.rust-kr.org/ch10-03-lifetime-syntax.html)

---

# 10. 완성된 프로그램

```rust
#[derive(Debug, Clone, Copy)]
struct Sample { stamp: f64, ax: f64, ay: f64, az: f64 }

impl Sample {
    fn magnitude(&self) -> f64 {
        (self.ax * self.ax + self.ay * self.ay + self.az * self.az).sqrt()
    }
}

fn parse_all(raw: &str) -> Vec<Sample> {
    let mut samples = Vec::new();
    for line in raw.lines() {
        let cols: Vec<&str> = line.split(',').collect();
        samples.push(Sample {
            stamp: cols[0].parse().unwrap(),
            ax: cols[1].parse().unwrap(),
            ay: cols[2].parse().unwrap(),
            az: cols[3].parse().unwrap(),
        });
    }
    samples
}

fn mean_magnitude(samples: &[Sample]) -> f64 {
    let mut sum = 0.0;
    for s in samples { sum += s.magnitude(); }
    sum / samples.len() as f64
}

fn worst_sample(samples: &[Sample]) -> &Sample {
    let mut worst = &samples[0];
    for s in samples {
        if s.magnitude() > worst.magnitude() { worst = s; }
    }
    worst
}

fn find_gaps(samples: &[Sample], expected_dt: f64) -> Vec<(f64, f64)> {
    let mut gaps = Vec::new();
    for pair in samples.windows(2) {
        let dt = pair[1].stamp - pair[0].stamp;
        if dt > expected_dt * 1.5 { gaps.push((pair[0].stamp, dt)); }
    }
    gaps
}

fn remove_gravity(samples: &mut Vec<Sample>, g: f64) {
    for s in samples.iter_mut() { s.az -= g; }
}

fn main() {
    let raw = "\
0.000,0.01,-0.02,9.79
0.005,0.02,-0.01,9.81
0.010,0.15,0.30,10.90
0.015,0.01,0.00,9.80
0.045,-0.02,0.01,9.78";

    let mut samples = parse_all(raw);

    println!("샘플 {}개", samples.len());
    println!("평균: {:.3}", mean_magnitude(&samples));

    let w = worst_sample(&samples);
    println!("최대: {:.3} at t={:.3}", w.magnitude(), w.stamp);

    for (t, dt) in find_gaps(&samples, 0.005) {
        println!("간격 이상: t={t:.3} 에서 {dt:.3}s");
    }

    remove_gravity(&mut samples, 9.81);
    println!("중력 제거 후 평균: {:.3}", mean_magnitude(&samples));
}
```

```text
샘플 5개
평균: 10.017
최대: 10.905 at t=0.010
간격 이상: t=0.015 에서 0.030s
중력 제거 후 평균: 0.249
```

새로 나온 것 두 가지를 짚어 둔다.

```rust
fn find_gaps(...) -> Vec<(f64, f64)>      // 튜플. 이름 없이 값 두 개를 묶는다
for (t, dt) in find_gaps(...)             // 구조 분해로 꺼낸다
```

`parse_all`이 `Vec<Sample>`을 **값으로 반환**하는 것도 눈여겨볼 만하다.
함수 안에서 만든 것을 밖으로 내보내는 것은 소유권을 **넘겨주는** 것이라 문제가 없다.

---

# 11. 시그니처만 보고 판단하기

Rust 함수는 시그니처만 봐도 무엇을 하는지 상당 부분 알 수 있다.

```rust
fn parse_all(raw: &str) -> Vec<Sample>
//           빌려서 읽기만 함      새로 만들어 소유권을 준다

fn mean_magnitude(samples: &[Sample]) -> f64
//                빌려서 읽기만 함. 원본 안 바뀜

fn worst_sample(samples: &[Sample]) -> &Sample
//              빌림                   입력 안의 것을 가리킨다
//                                     → 입력이 살아 있어야 유효

fn remove_gravity(samples: &mut Vec<Sample>, g: f64)
//                        원본을 고친다!         Copy 라 그냥 값
//                반환값 없음 → 부수효과가 목적이다

fn consume(samples: Vec<Sample>)
//                  가져가 버린다. 호출 후 원본 못 씀
```

**C++에서는 주석이나 문서를 봐야 알 수 있는 것이 타입에 적혀 있다.**
`const&`인지 `&`인지 값인지 헷갈릴 일이 없고, 컴파일러가 강제한다.

---

# 12. Mini Practice

```bash
cd /tmp/imu_stats
```

```text
1) mean_magnitude 의 &[Sample] 을 Vec<Sample> 로 바꾸고 cargo check
   → E0382. 에러 메시지가 어떤 수정을 제안하나?

2) remove_gravity 의 &mut 을 & 로 바꾸고 cargo check
   → error[E0596]: cannot borrow `*samples` as mutable,
                   as it is behind a `&` reference
   "& 로 빌렸으면서 고치려 한다"는 뜻이다

3) main 의 let mut samples 에서 mut 을 지우고 cargo check
   → E0596. 어느 줄을 가리키나?

4) worst_sample 이 반환한 참조를 오래 살려 본다:
       let w = worst_sample(&samples);
       remove_gravity(&mut samples, 9.81);
       println!("{:?}", w);
   → E0502. 왜 위험한지 §8 과 연결해서 설명해 보기

5) 아래를 넣고 무슨 일이 생기는지
       let mut v = vec![1.0, 2.0, 3.0];
       let first = &v[0];
       v.push(4.0);
       println!("{first}");
   → E0502. 같은 코드를 C++ 로 쓰면 어떻게 되나?

6) Sample 에서 Copy 를 지운다: #[derive(Debug, Clone)]
   → 놀랍게도 아무 에러도 안 난다. 왜일까?

7) 6번 상태에서 main 에 한 줄을 추가한다:
       let first = samples[0];
       println!("{:?}", first);
   → error[E0507]: cannot move out of index of `Vec<Sample>`
                   ... `Sample`, which does not implement the `Copy` trait
```

6번과 7번을 붙여서 보는 게 이 chapter의 마무리다.

```text
6번에 에러가 없는 이유
   이 프로그램은 Sample 을 값으로 꺼내 쓰는 곳이 한 군데도 없다.
   전부 &[Sample] 로 빌려서 읽거나 iter_mut() 으로 고친다.
   → Copy 가 필요 없었던 것이다

7번에서 터지는 이유
   samples[0] 은 Vec 안에서 값을 "꺼내오는" 것이다.
   Copy 면 복사되고, 아니면 move 인데
   Vec 의 일부만 move 로 빼낼 수는 없다 (나머지가 구멍 난 Vec 이 된다)
   → 컴파일러가 &samples[0] 또는 .clone() 을 제안한다
```

**참조로만 다루면 소유권 문제가 거의 안 생긴다**는 것이 6번의 교훈이고,
값을 꺼내는 순간 `Copy`인지 아닌지가 드러난다는 것이 7번의 교훈이다.

4번과 7번이 특히 중요하다.
**Rust 배우기 = 컴파일러와 대화하기**라는 말이 무슨 뜻인지 알게 된다.

---

# 13. 오늘의 핵심

```text
                    값을 함수에 넘길 때

   fn f(x: Vec<T>)        move    가져간다. 호출 후 원본 못 씀
   fn f(x: &Vec<T>)       borrow  빌려서 읽는다
   fn f(x: &[T])          borrow  ← 읽기 전용이면 이게 기본
   fn f(x: &mut Vec<T>)   borrow  빌려서 고친다

                    동시에 가질 수 있는 것

   &T  &T  &T  &T         불변 참조는 여러 개 OK
   &mut T                 가변 참조는 딱 하나
   &T + &mut T            ✗ 섞을 수 없다

                    왜?

   Vec 이 재할당되면 옛 주소는 해제된다
   C++ 이면 dangling. Rust 는 컴파일 거부
```

---

# 14. 반드시 구분할 것

```text
move  ≠  copy
   Copy 가 구현된 타입만 복사된다 (f64, Sample)
   Vec, String 은 move

&  ≠  &mut
   읽기 / 쓰기. 호출부에도 &mut 을 적는다

&Vec<T>  ≠  &[T]
   후자가 더 많은 것을 받는다. 읽기 전용이면 &[T]

&String  ≠  &str
   같은 이유로 &str

iter()  vs  iter_mut()  vs  into_iter()
   빌려 읽기 / 빌려 고치기 / 가져가기

clone()  ≠  borrow
   clone 은 진짜 복사다. 큰 데이터에 습관적으로 쓰면 안 된다

컴파일 에러  ≠  까다로움
   C++ 이면 런타임에 터졌을 것을 미리 잡는 것

참조 반환이 되는 이유
   입력이 살아 있는 동안만 유효하다고 컴파일러가 묶어 둔다
```

---

# 15. Chapter 연결

```text
Chapter 7
프로그램 한 덩어리 — 소유권 문제가 안 생긴다

Chapter 8  ← 여기
함수로 쪼갠다 → move, borrow, &mut, 슬라이스

Chapter 9
raw 문자열 대신 진짜 파일을 읽는다
→ unwrap() 이 걷히고 Result, Option, ?, match 가 나온다

Edge Computing Chapter 15
GC 없이 실시간 보장 — Rust 를 로봇에 쓰는 이유
```
