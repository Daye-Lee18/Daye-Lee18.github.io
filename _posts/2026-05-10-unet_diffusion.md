---
layout: post
title: U-Net 
date: 2026-05-10
description: Lerobot의 diffusion policy에서 쓰인 U-Net 구조에 대해 공부. 
tags: [LeRobot, Diffusion, U-Net, ConvTranspose1d]
featured: true
categories: study
toc:
  sidebar: left
---

# ConditionalUnet1D 구조 (Diffusion Policy)

lerobot의 diffusion policy 학습을 하며, chuck_size 를 마다 학습하여 모델을 성능을 비교하였다. u-net이 입력으로 받을 수 있는 chuck_size는 4의 배수여야하는데, 그 이유에 대해 알아보자. 

## 1. 전체 흐름 테이블 (chunk_size=T, action_dim=7)

| 단계 | 모듈 | 채널 | T (예: 20) | skip 저장? |
|------|------|------|------------|-----------|
| 입력 | — | 7 | 20 | — |
| **Down 0** | ResBlock×2 + Conv1d(stride=2) | 7→256 | 20→**10** | h에 T=20 저장 |
| **Down 1** | ResBlock×2 + Conv1d(stride=2) | 256→512 | 10→**5** | h에 T=10 저장 |
| **Down 2** | ResBlock×2 + Identity | 512→1024 | 5→**5** | h에 T=5 저장 |
| **Mid** | ResBlock×2 | 1024→1024 | 5 | — |
| **Up 0** | cat(h.pop=T5) + ResBlock×2 + ConvTranspose1d(stride=2) | 1024×2→512 | 5→**10** | — |
| **Up 1** | cat(h.pop=T10) + ResBlock×2 + ConvTranspose1d(stride=2) | 512×2→256 | 10→**20** | — |
| **Final** | Conv1d(1×1) | 256→7 | 20 | — |

> h에 저장된 T=20 (Down 0)은 Up이 2개뿐이라 사용되지 않음

---

## 2. 왜 chunk_size가 4의 배수여야 하는가

`Downsample1d` = `Conv1d(kernel=3, stride=2, padding=1)`

$$T_{out} = \left\lfloor\frac{T_{in} - 1}{2}\right\rfloor + 1$$

T가 **짝수**일 때만 정확히 T/2가 됩니다:

```
T=20 → ⌊19/2⌋+1 = 10  (짝수) → ⌊9/2⌋+1 = 5
T=30 → ⌊29/2⌋+1 = 15  (홀수!) → ⌊14/2⌋+1 = 8
```

`Upsample1d` = `ConvTranspose1d(kernel=4, stride=2, padding=1)`

$$T_{out} = (T_{in} - 1) \times 2 - 2 + 4 = 2 \times T_{in}$$

항상 정확히 2배. Up에서 skip과 크기가 맞으려면 **Down 결과가 짝수여야** 합니다:

```
Down이 2번 실질 발생 → T, T/2, T/4 가 모두 정수여야 함
  → T/2 정수 : T가 2의 배수
  → T/4 정수 : T가 4의 배수
```

| chunk_size | T/2 | T/4 | 결과 |
|------------|-----|-----|------|
| 20 | 10 | 5 | ✓ |
| 30 | 15 | 7.5 | ✗ (Down 후 홀수 → Up 크기 불일치) |
| 32 | 16 | 8 | ✓ |
| 24 | 12 | 6 | ✓ |

권장 값: **20, 24, 28, 32, 36, 40 ...**  (4의 배수)

> `diffusion_utils.py`에 크기 트리밍 패치가 적용되어 있어 4의 배수가 아니어도 동작하지만,  
> 의도한 skip connection이 정확히 복원되려면 4의 배수를 사용하는 것이 바람직합니다.

---

## 3. Conv2d vs Conv1d — 핵심 구조

> **Key**
>
> Conv2d에서 `(C_out, C_in, kH, kW)` 짜리 커널이 `H×W` 위를 슬라이딩하는 것처럼,  
> Conv1d에서는 `(C_out, C_in, k)` 짜리 커널이 `L` 위를 슬라이딩합니다.  
> **구조는 완전히 동일하고, 공간 차원만 2개 → 1개로 줄어든 것입니다.**

<details>
<summary>세부 내용 보기</summary>

### 입력/출력 shape

| | Conv2d | Conv1d |
|---|---|---|
| 입력 | (B, C_in, **H, W**) | (B, C_in, **L**) |
| 출력 | (B, C_out, **H_out, W_out**) | (B, C_out, **L_out**) |
| 커널 전체 | **(C_out, C_in, kH, kW)** | **(C_out, C_in, k)** |

### 커널이 동작하는 방식

```
입력: (B, C_in=3, H, W)

커널 #0   (C_in=3, kH, kW)  → H×W 슬라이딩 → 출력 채널 0   (H_out, W_out)
커널 #1   (C_in=3, kH, kW)  → H×W 슬라이딩 → 출력 채널 1   (H_out, W_out)
...
커널 #63  (C_in=3, kH, kW)  → H×W 슬라이딩 → 출력 채널 63  (H_out, W_out)

출력: (B, C_out=64, H_out, W_out)
```

- **C_in** → 커널 1개가 입력 채널 전부를 동시에 봄 (depth 방향 합산)
- **C_out** → 서로 다른 패턴을 학습하는 커널의 수 → 출력 채널 수

### 출력 크기 공식

**Conv1d (다운샘플 방향)**

$$L_{out} = \left\lfloor \frac{L_{in} + 2P - K}{S} \right\rfloor + 1$$

**ConvTranspose1d (업샘플 방향)**

$$L_{out} = (L_{in} - 1) \times S - 2P + K$$

</details>

---

## 4. ConvTranspose1d — 역할과 사용 시점

> **Key**
>
> Conv1d가 공간을 **축소**하는 방향이라면,  
> ConvTranspose1d는 입력 사이에 공백을 삽입한 뒤 Conv를 적용해 공간을 **확장**합니다.  
> Bilinear/nearest-neighbor와 달리 **가중치를 학습**하므로 "어떻게 복원할지"를 데이터로부터 배웁니다.

<details>
<summary>세부 내용 보기</summary>

```
일반 Conv (stride=2):     [a, b, c, d] → [x, y]        (축소)
ConvTranspose (stride=2): [x, y]        → [a, b, c, d]  (복원)
```

"Transposed"라는 이름은 Conv를 행렬 W로 표현할 때, ConvTranspose가 W^T 를 곱하는 연산이기 때문입니다.

### ConvTranspose1d/2d가 쓰이는 곳

| 용도 | 이유 |
|------|------|
| UNet decoder (Diffusion Policy) | skip feature와 합치기 전 해상도 복원 |
| GAN Generator | 잠재 벡터 → 고해상도 이미지/시퀀스 생성 |
| VAE Decoder | latent z → 원본 크기 복원 |
| 이미지 초해상도 | 저해상 → 고해상 |
| Semantic segmentation decoder | 픽셀 단위 예측을 위한 업샘플 |

</details>
