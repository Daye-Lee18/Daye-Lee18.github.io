---
layout: post
title: Transformer
date: 2026-05-11
description: Tansformer study 
tags: [Attention, Transformer, FFN, ]
featured: false
categories: study
toc:
  sidebar: left
---

# Transformer Dimension Guide (ACT-VQ 기준)

## 설정값 (train_act_vq.sh 기본값)

| 인자              | 값    |
|------------------|-------|
| `hidden_dim`     | 512   |
| `dim_feedforward`| 3200  |
| `chunk_size`     | 20    |
| `vq_class`       | 16    |
| `vq_dim`         | 32    |

---

## 1. hidden_dim (= d_model = 512)

**네트워크 전체를 흐르는 "토큰 벡터의 크기".**

```
입력 토큰 → [512] → (모든 레이어) → [512] → 최종 출력
```

### 어디서 쓰이나

| 위치                     | 입력 dim  | 출력 dim  |
|--------------------------|-----------|-----------|
| Joint embedding (Linear) | 7 (DOF)   | **512**   |
| Image feature projection | CNN 출력   | **512**   |
| Q, K, V projection       | **512**   | **512**   |
| Attention output         | **512**   | **512**   |
| LayerNorm                | **512**   | **512**   |
| FFN 입력                  | **512**   | 3200      |
| FFN 출력                  | 3200      | **512**   |
| Action head (최종 Linear) | **512**   | 7 (DOF)   |

> hidden_dim 을 바꾸면 **모델 전체**가 영향을 받는다.

---

## 2. dim_feedforward (= d_ff = 3200)

**FFN(Feed-Forward Network) 안의 중간 팽창 크기. FFN 밖에서는 안 쓰인다.**

### FFN 구조

```
x: [B, T, 512]
        │
   Linear(512 → 3200)   ← dim_feedforward
        │
      ReLU
        │
   Linear(3200 → 512)   ← dim_feedforward → 다시 hidden_dim으로
        │
x_out: [B, T, 512]
```

### 배율 관계

```
dim_feedforward / hidden_dim = 3200 / 512 ≈ 6.25×

일반적인 범위: 4× ~ 8×
  - BERT: 3072 / 768 = 4×
  - GPT-3: 4× (일반적)
  - 이 설정: 6.25×  ← 표현력을 높이기 위해 크게 설정
```

> dim_feedforward 를 바꿔도 **FFN 내부 파라미터 수만** 달라지고,
> 토큰 벡터 크기(512)는 그대로다.

---

## 3. 레이어 한 개의 dim 흐름 (Encoder 기준)

```
입력: [B, T, 512]
        │
┌───────┴────────────────────────────────────────────────┐
│  Self-Attention                                         │
│    Q = Linear(512→512),  K = Linear(512→512)           │
│    V = Linear(512→512)                                  │
│    score = softmax( Q·Kᵀ / √512 ) · V   → [B,T,512]   │
│    out   = Linear(512→512)               → [B,T,512]   │
├────────────────────────────────────────────────────────┤
│  Add & Norm:  LayerNorm([B,T,512])                      │
├────────────────────────────────────────────────────────┤
│  FFN                                                    │
│    Linear(512 → 3200) → ReLU                           │
│    Linear(3200 → 512)               → [B,T,512]        │
├────────────────────────────────────────────────────────┤
│  Add & Norm:  LayerNorm([B,T,512])                      │
└───────┬────────────────────────────────────────────────┘
        │
출력: [B, T, 512]    ← 입력과 shape 동일
```

Multi-head attention 의 경우 (num_heads = h):
```
각 head의 dim = hidden_dim / num_heads = 512 / 8 = 64
  → d_k = d_v = 64
  → score = softmax( Q·Kᵀ / √64 )   (분모는 √d_k)
  → 모든 head concat 후 Linear(512→512)
```

---

## 4. chunk_size 와 query embedding

ACT-VQ 에서 chunk_size 는 **Decoder의 query 토큰 수**로 쓰인다.

```
chunk_size = 20 이면:
  query embedding: [20, 512]  (학습 가능한 파라미터)
        │
  Decoder cross-attention  (K, V 는 Encoder 출력)
        │
  Decoder 출력: [20, 512]
        │
  Action head (Linear 512→7) × 20
        │
  예측 action: [20, 7]   ← 앞으로 20 스텝의 joint 값
```

chunk_size 를 바꾸면 query 토큰 수만 달라지고, hidden_dim/dim_feedforward 는 그대로다.

---

## 5. 파라미터 수 rough 계산 (레이어 1개)

| 구성요소        | 파라미터 수                          |
|----------------|--------------------------------------|
| Q, K, V Linear | 3 × (512 × 512) = **786,432**        |
| Attention out  | 512 × 512 = **262,144**              |
| FFN Linear1    | 512 × 3200 = **1,638,400**           |
| FFN Linear2    | 3200 × 512 = **1,638,400**           |
| LayerNorm ×2   | 2 × 512 × 2 = **2,048**             |
| **합계 (1레이어)** | **≈ 4.3M**                       |

> FFN 이 Attention 의 약 4배 파라미터를 가진다.
> dim_feedforward 를 키우면 FFN 파라미터가 선형으로 증가한다.
