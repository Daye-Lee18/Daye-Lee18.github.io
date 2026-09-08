---
layout: study-topic
title: Rust
description: 숲에서 나무로 — workspace, package, crate, module과 Cargo 빌드 구조부터.
permalink: /study/rust/
topic_index: true
---

문법보다 **구조**를 먼저 본다.
`Cargo.toml`이 왜 두 개인지, `crate`가 `[workspace].members`에 들어가는 건지,
`cargo build`가 무엇을 어디에 만드는지를 먼저 정리하고 언어로 내려간다.

```text
숲 ──────────────────────────────────────────▶ 나무

1부 · 구성과 빌드
  Ch 1   workspace / package / crate / module    전체 지도
  Ch 2   Cargo.toml 읽는 법                      설정 파일
  Ch 3   cargo 명령어와 target/                  빌드가 하는 일
  Ch 4   의존성과 Cargo.lock                     바깥과 연결
  Ch 5   module 시스템 (mod / pub / use)         crate 안으로
  Ch 6   ROS 2 저장소 안에서 Rust 쓰기            실제 프로젝트

2부 · 언어  ─ 프로그램 하나를 계속 고쳐 가며
  Ch 7   돌아가는 프로그램 하나 읽기              struct, impl, Vec, for
  Ch 8   함수로 쪼개다가 소유권을 만나기           move, &, &mut, 슬라이스
  Ch 9   진짜 파일을 읽으며 unwrap() 걷어내기      Result, Option, ?, match
```

## 2부의 공부 방식

문법을 하나씩 배우고 예제를 보는 순서가 아니다.
**돌아가는 프로그램을 먼저 두고, 거기 실제로 쓰인 것만 설명한다.**

```text
Chapter 7    IMU 로그를 읽어 통계를 내는 프로그램 (한 덩어리)
                     │  함수로 쪼갠다
Chapter 8    같은 프로그램 → 소유권 에러를 만난다
                     │  진짜 파일을 읽게 한다
Chapter 9    같은 프로그램 → unwrap() 을 걷어낸다
```

Chapter 9까지 오면 프로그램은 실제 CSV 파일을 읽고,
깨진 줄·빈 파일·없는 파일을 전부 사람이 읽을 수 있는 메시지와
종료 코드로 처리한다. 패닉이 하나도 남지 않는다.

새 예제를 만들지 않는다. **한 프로그램이 계속 자란다.**
안 쓰인 문법은 안 나오고, 필요해지면 그때 나온다.

교재는 [Rust 프로그래밍 언어 (한국어판)](https://doc.rust-kr.org/title-page.html)을 참고했고,
각 절 끝에 대응하는 장을 링크해 두었다.
