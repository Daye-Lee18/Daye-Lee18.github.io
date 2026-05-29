---
layout: post
title: V-A Terminology
date: 2026-05-17
description: V-A (Vision-Action) 모방학습 정책 연구에서 자주 쓰이는 용어 한 페이지 정리 — 모방학습, Policy, CVAE, ACT, ACT-VQ, Diffusion Policy, Temporal Aggregation 등.
tags: [imitation-learning, ACT, ACT-VQ, "Diffusion Policy", CVAE, policy, robotics]
featured: false
categories: study
toc:
  sidebar: left
---

> 발표 직전 빠르게 훑어보기 위한 V-A (Vision-Action) 연구 용어 사전.  
> 정의 → 본 연구에서의 의미 → 자주 헷갈리는 점 순으로 정리.

---

## 1. 모방학습 / Imitation Learning 일반

### 모방학습 (Imitation Learning, IL)

전문가(human)가 수행한 시연(demonstration)을 모사하도록 정책을 학습하는 패러다임. 보상 함수가 필요 없다는 점에서 **강화학습(RL)** 과 구분된다.

- **Behavior Cloning (BC)**: 가장 단순한 IL. 시연 데이터에 대해 $\pi(a \mid o)$ 를 **지도학습(supervised regression)** 으로 학습. 분포 외 상태(OOD)에서 covariate shift로 실패하는 게 약점.
- **DAgger (Dataset Aggregation)**: BC의 covariate shift를 해소하기 위해 _학습된 policy로 굴려보고 → 실패 상태에서 전문가가 라벨링 → 데이터셋 추가_ 를 반복.
- 본 연구의 ACT/Diffusion Policy 모두 **BC 계열의 발전형**.

### Demonstration / Trajectory / Episode

| 용어                    | 의미                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Demonstration**       | 전문가가 1회 수행한 시연 그 자체                                                                            |
| **Trajectory** $$\tau$$ | 시간순으로 정렬된 $$(o_t, a_t)$$ sequence                                                                   |
| **Episode**             | Task 1회 수행 단위 = 1 trajectory. 본 연구 기준: _AMR 도착 → 주물 pick → unloading 완료_ (~17 s, ~100 step) |

{: .table .table-sm .table-striped}

### Teleoperation (원격 조작)

사람이 **Leader arm**을 움직이면 **Follower arm**이 동일한 관절각으로 따라가는 시연 수집 방식. ALOHA, GELLO 등이 대표.

### Policy ($\pi$)

**Observation → Action 매핑 함수** $\pi: o \mapsto a$.

- 본 연구에서 $o$ = (multi-view image, qpos), $a$ = (next qpos or qpos chunk, gripper).
- _"policy = decoder"_ 라고 흔히 말하지만, 엄밀히는 **추론 시점의 ACT만 그렇고**, 학습 시점이나 Diffusion Policy에서는 다름.

### Closed-loop vs Open-loop

| 구분            | 의미                                                | 본 연구                   |
| --------------- | --------------------------------------------------- | ------------------------- |
| **Open-loop**   | 처음 한 번 추론한 action chunk를 끝까지 그대로 실행 | ACT chunk mode            |
| **Closed-loop** | 매 step 관측 받아서 action 다시 추론                | Temporal Aggregation mode |

{: .table .table-sm .table-striped}

---

## 2. V-A / VLA 패러다임

### V-A (Vision-Action)

**시각 관측 → 행동** 으로 곧장 매핑하는 end-to-end 정책. 본 연구의 핵심 paradigm.

$$\pi_\theta(a_t \mid I_t^{top}, I_t^{side}, I_t^{wrist}, q_t)$$

### VLA (Vision-Language-Action)

V-A에 **language instruction** 조건이 추가된 형태. 대표: **RT-2, OpenVLA, π₀**.

$$\pi_\theta(a_t \mid I_t, q_t, \ell)$$

- $\ell$ = "pick the white casting" 같은 자연어 명령
- task semantic을 language로 anchor → generalization 강함
- 본 연구의 향후 확장 방향.

### Observation Space / Action Space

- **Observation**: $o_t = (I_t^{top}, I_t^{side}, I_t^{wrist}, q_t)$ — 3시점 image + 7-dim qpos (6 joint + 1 gripper).
- **Action**: $a_t$ — 다음 시점의 qpos (또는 chunk 형태로 $a_{t:t+k}$).

---

## 3. ACT (Action Chunking with Transformers)

> Tony Z. Zhao et al., _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_ (RSS 2023).

### 핵심 아이디어

1. **Action Chunking**: 매 step 1개씩 action을 예측하는 대신 **k step 묶음(chunk)** 을 한 번에 예측 → 누적 오차 ↓, smooth motion.
2. **CVAE 구조**: behavior의 multi-modality를 latent $z$로 다룸.
3. **Temporal Aggregation**: 같은 시점에 대해 여러 chunk의 예측을 가중평균 → noise 감소.

### CVAE (Conditional Variational Autoencoder) in ACT

| 모듈                   | 입력                             | 출력                       | 사용 시점       |
| ---------------------- | -------------------------------- | -------------------------- | --------------- |
| **Style Encoder**      | `[CLS]` + action sequence + qpos | latent $$z$$ (Gaussian)    | 학습 only       |
| **Decoder (= policy)** | image features + qpos + $$z$$    | action chunk $$a_{t:t+k}$$ | 학습 + **추론** |

{: .table .table-sm .table-striped}

- 추론 시 $$z = \mathbf{0}$$ (prior에서 sample) → 사실상 decoder만 작동 ⇒ 이때만 _policy ≈ decoder_.
- 학습 loss = **L1 reconstruction loss** + $$\beta \cdot D_{KL}(q_\phi(z\mid x) \,\Vert\, \mathcal{N}(0,I))$$.

### 주요 하이퍼파라미터

| 인자                 | 의미                                                 | 본 연구 값     |
| -------------------- | ---------------------------------------------------- | -------------- |
| `chunk_size` ($$k$$) | 한 번에 예측하는 action 개수                         | 20, 28, 40, 52 |
| `num_queries`        | 한 episode 당 추론 횟수 (= episode_len / chunk_size) | 3 ~ 6          |
| `hidden_dim`         | transformer token dim                                | 512            |
| `dim_feedforward`    | FFN 내부 dim                                         | 3200           |

{: .table .table-sm .table-striped}

### Chunk Mode vs Temporal Aggregation Mode

```
[Chunk mode]                       [Temp-Agg mode]
t=0  → 예측 a[0..k-1] 전부 실행      t=0  → 예측 a[0..k-1]
t=k  → 예측 a[k..2k-1] 전부 실행     t=1  → 새로 예측 a'[1..k]
...                                       → a_1 위치는 a[1], a'[1] 가중평균
                                    t=2  → 또 새로 예측 ...
```

- **Chunk mode**: open-loop, 빠름, 단 환경 변화에 둔감.
- **Temp-Agg mode**: closed-loop, smooth, 단 매 step 추론이라 느림 (본 연구에선 ~33 s 지연).

---

## 4. ACT-VQ

ACT의 latent $z$를 **continuous Gaussian → discrete codebook** 으로 바꾼 변형.

### VQ (Vector Quantization)

- Continuous latent를 **K개의 codebook vector** 중 가장 가까운 것으로 양자화.
- Codebook: $\mathcal{C} = \{e_1, \dots, e_K\}, \quad e_i \in \mathbb{R}^d$.
- 본 연구: `vq_class=16`, `vq_dim=32`.

### 왜 VQ?

- Continuous latent의 **posterior collapse** 회피.
- Behavior의 **discrete mode** (예: "왼쪽으로 잡기" vs "오른쪽으로 잡기")를 명시적으로 학습.

### Loss

$$\mathcal{L}_{VQ} = \mathcal{L}_{recon} + \underbrace{\|sg[z_e] - e\|^2}_{\text{codebook loss}} + \beta\underbrace{\|z_e - sg[e]\|^2}_{\text{commitment loss}}$$

- $sg[\cdot]$ = stop-gradient.

---

## 5. Diffusion Policy

> Cheng Chi et al., _Diffusion Policy: Visuomotor Policy Learning via Action Diffusion_ (RSS 2023).

### 핵심 아이디어

Action chunk를 **노이즈에서 시작해서 K번 denoising** 으로 생성. 이미지 생성의 DDPM을 action 생성에 적용.

$$a^{(0)} \sim \mathcal{N}(0, I) \to a^{(1)} \to \dots \to a^{(K)} = \hat{a}$$

각 step에서 noise predictor $\epsilon_\theta(a^{(k)}, k, o)$ 가 noise를 추정.

### 학습 목적

$$\mathcal{L} = \mathbb{E}_{a, \epsilon, k}\!\left[\|\epsilon - \epsilon_\theta(a^{(k)}, k, o)\|^2\right]$$

### ACT와의 차이

| 항목               | ACT                           | Diffusion Policy           |
| ------------------ | ----------------------------- | -------------------------- |
| 생성 방식          | 1-step forward (CVAE decoder) | K-step iterative denoising |
| Latent 구조        | Gaussian $$z$$                | Noise schedule             |
| Multi-modal action | $$z$$ 통해 표현               | 자연스럽게 표현            |
| 추론 속도          | 빠름                          | 느림 (K iteration)         |
| 본 연구 결과       | succ 90 %                     | succ **0 %** (이 setup)    |

{: .table .table-sm .table-striped}

---

## 6. 평가 / 실패 분석 용어

### Success Rate (Succ Rate)

N rollouts 중 task 성공 횟수 비율. 본 연구는 10 rollouts 기준.

### Generalization (일반화)

학습 분포 **밖**의 조건에서 성능 유지 능력. 본 연구에선 2축으로 검증:

- **주물 종류 변경** → 일반화 ✓
- **AMR 위치 변경** → 일반화 ✗ (학습 분포 밖일 때)

### OOD (Out-of-Distribution)

입력이 학습 데이터 분포 **밖**에 있는 상황. **본 연구 발견 1**: 추론된 action이 _물리적으로는 도달 가능한 위치임에도_ 학습 데이터셋 내의 action 분포 반경 안에만 머무름 → 결국 실패.

### Temporal Overfitting

**본 연구 발견 2.** 모델이 image보다 **timestep $t$ 자체**에 더 의존하는 현상. 동일한 wrist image에 대해 $t=29$ vs $t=93$에서 상이한 action 출력. Transformer의 positional encoding이 image feature보다 더 강한 signal로 작동한 것으로 추정.

### Covariate Shift

BC 학습 시 _학습 시 본 상태 분포_ ↔ _실제 rollout에서 마주치는 상태 분포_ 가 어긋나서 누적 오차로 실패하는 현상. DAgger / Recovery demonstration이 해결책.

---

## 7. 본 연구 setup 약어 모음

| 약어               | 풀이                                               |
| ------------------ | -------------------------------------------------- |
| **AMR**            | Autonomous Mobile Robot (PinkyPro 사용)            |
| **DoF**            | Degree of Freedom (MyCobot280 = 6 DoF)             |
| **qpos**           | Joint position vector (7-dim: 6 joint + 1 gripper) |
| **FPS / Hz**       | 데이터 수집 / 추론 주기 (6 Hz → 목표 30 Hz)        |
| **HDF5**           | dataset 저장 포맷                                  |
| **LeRobot format** | HuggingFace robotics 표준 포맷 (향후 공개용)       |

{: .table .table-sm .table-striped}

---

## 8. 자주 받을 만한 질문 대비

**Q. "Policy = decoder 인가요?"**
→ ACT 추론 시점 한정으로는 맞음. 학습 시엔 encoder도 정책 학습에 관여하고, Diffusion Policy에서는 decoder가 아니라 denoising network임.

**Q. "chunk_size를 왜 hyperparameter로 봤나요?"**
→ chunk가 클수록 한 번에 더 멀리 예측 → GPU 메모리 ↑, 추론 횟수 ↓. 즉 **성능 vs 자원 trade-off**의 핵심 knob.

**Q. "succ rate 90 % 인데 왜 한 번 실패?"**
→ Gripper 여는 timing이 살짝 어긋남 (정확한 grasp 시점은 데이터에서 좁은 분포라 sensitive).

**Q. "Diffusion Policy가 왜 0 %?"**
→ Diffusion은 보통 더 많은 데이터·step이 필요. 25 episodes · chunk=20 · 1800 step 학습 setup에선 underfit으로 보임. Episode 수↑ / DDIM step↑ ablation 필요.

**Q. "Temporal Overfitting을 어떻게 해결할 계획?"**
→ ① timestep embedding dropout/제거, ② episode 길이 randomization, ③ pretrained vision encoder(DINOv2 등)로 image feature 표현력 ↑, ④ 궁극적으로 VLA로 language anchor.

---

## References

- Zhao, T. Z. et al. _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_ (ACT). RSS 2023.
- Chi, C. et al. _Diffusion Policy: Visuomotor Policy Learning via Action Diffusion_. RSS 2023.
- Lee, S. et al. _Behavior Generation with Latent Actions_ (VQ-BeT). ICML 2024.
- Brohan, A. et al. _RT-2: Vision-Language-Action Models_. 2023.
- Kim, M. J. et al. _OpenVLA: An Open-Source Vision-Language-Action Model_. CoRL 2024.
