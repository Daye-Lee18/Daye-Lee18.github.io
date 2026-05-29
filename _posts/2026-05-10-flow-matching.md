---
layout: post
title: Flow Matching (FM) 요약
date: 2026-05-10
description: Flow Matching의 핵심 개념과 수학적 구조, 그리고 로봇 제어(Pi Model)로의 응용을 정리합니다.
tags: [FlowMatching, GenerativeModel, Robotics, "Pi Model"]
featured: true
categories: study
toc:
  sidebar: left
---

## 1. Flow Matching (FM) 핵심 개념

Flow Matching은 데이터를 생성하는 확률 경로(Probability Path)를 정의하고, 이를 유도하는 **벡터장(Vector Field)**을 직접 학습하는 생성 모델입니다. Diffusion 모델과 유사하지만, 노이즈에서 데이터로 가는 경로를 더 직접적이고 효율적으로 설계할 수 있습니다.

### 핵심 구성 요소

- **벡터장 $v_t(x)$**: 시간 $t$에 따라 샘플 $x$가 이동해야 할 방향과 속도를 정의합니다.
- **흐름(Flow) $\psi_t(x)$**: 벡터장에 의해 정의되는 궤적으로, $\frac{d}{dt}\psi_t(x) = v_t(\psi_t(x), t)$를 만족합니다.
- **확률 경로 $p_t(x)$**: $p_0$(노이즈 분포)에서 $p_1$(데이터 분포)로 변하는 연속적인 분포의 집합입니다.

---

## 2. 수학적 구조 및 학습 방법

### Conditional Flow Matching (CFM)

실제 데이터 분포의 벡터장을 직접 구하는 것은 어렵기 때문에, 개별 데이터 포인트 $x_1$에 조건화된 **조건부 벡터장**을 학습합니다.

- **선형 보간 경로 (Linear Interpolation)**: 가장 흔히 쓰이는 경로 설정입니다.
  $$x_t = (1-t)x_0 + t x_1$$
- **조건부 벡터장 $u_t(x \mid x_1)$**: 위 경로를 미분하면 얻어지는 목표 벡터장입니다.
  $$u_t(x_t \mid x_1) = \frac{d}{dt}x_t = x_1 - x_0$$

### 목적 함수 (Loss Function)

신경망 $v_\theta(x, t)$가 목표 벡터장 $u_t$를 모사하도록 학습합니다.
$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

> **Tip:** Diffusion 모델과 달리 학습 시 ODE Solver를 거칠 필요가 없으며, 목적 함수가 단순한 MSE 형태이므로 학습이 매우 안정적입니다.

---

## 3. 코드 응용 (Implementation Logic)

PyTorch 스타일의 핵심 로직입니다.

```python
import torch

def flow_matching_loss(v_net, x_1):
    # 1. 시간 t 샘플링 (0 ~ 1)
    t = torch.rand(x_1.shape[0], 1, device=x_1.device)

    # 2. 베이스 분포(노이즈) x_0 생성
    x_0 = torch.randn_like(x_1)

    # 3. 선형 보간을 통한 x_t 생성
    x_t = (1 - t) * x_0 + t * x_1

    # 4. 목표 벡터장 (x_1 - x_0)
    target_v = x_1 - x_0

    # 5. 모델 예측 및 손실 계산
    pred_v = v_net(x_t, t)
    loss = torch.mean((pred_v - target_v) ** 2)

    return loss
```

---

## 4. 로봇 공학 응용: Pi Model

Flow Matching은 로봇의 **Policy Inference (Pi Model)**에서 다봉성(Multi-modality)을 가진 행동 분포를 모델링하는 데 사용됩니다.

- **Pi Model ($\pi(a \mid s)$)**: 주어진 상태 $s$에서 최적의 행동 $a$를 생성하는 모델입니다.
- **장점**: Diffusion Policy에 비해 추론 시 적은 step(Iteration)으로도 고품질의 행동 시퀀스를 생성할 수 있어, 실시간 제어가 중요한 로봇 시스템에 적합합니다.
- **Flow**: 노이즈 $a_0$로부터 시작하여 벡터장을 따라 적분함으로써 최종 행동 $a_1$을 도출합니다.

---

## 5. 연습 문제

### Q1. 조건부 벡터장 유도

선형 보간 경로가 다음과 같이 정의된다:

$$x_t = (1-t)x_0 + t x_1, \qquad t \in [0, 1]$$

**(a)** $x_t$를 시간 $t$에 대해 미분하여 조건부 벡터장 $u_t(x_t \mid x_1)$를 구하라.

**(b)** $t=0$과 $t=1$에서의 $u_t$ 값이 각각 $x_1 - x_0$로 동일한 이유를 설명하라.

<details>
<summary>정답 보기</summary>

**(a)**

$$u_t(x_t \mid x_1) = \frac{d}{dt}x_t = \frac{d}{dt}\left[(1-t)x_0 + tx_1\right] = -x_0 + x_1 = x_1 - x_0$$

**(b)** 경로가 **선형**이므로 기울기(속도)가 $t$에 무관하게 상수 $x_1 - x_0$다. 직선 위를 일정한 속도로 이동하기 때문에 출발점($t=0$)과 도착 직전($t=1$) 모두 같은 방향·크기의 벡터장을 가진다.

</details>

---

### Q2. 손실 함수 계산 (수치 예시)

$x_0 = -1$, $x_1 = 3$, $t = 0.25$일 때:

**(a)** $x_t$를 계산하라.

**(b)** 목표 벡터장 $u_t(x_t \mid x_1)$를 계산하라.

**(c)** 모델 예측이 $v_\theta(x_t, t) = 3.5$일 때, CFM 손실 $\| v_\theta - u_t \|^2$를 계산하라.

<details>
<summary>정답 보기</summary>

**(a)**

$$x_{0.25} = (1 - 0.25)(-1) + 0.25(3) = -0.75 + 0.75 = 0$$

**(b)**

$$u_t = x_1 - x_0 = 3 - (-1) = 4$$

**(c)**

$$\|v_\theta - u_t\|^2 = |3.5 - 4|^2 = (-0.5)^2 = 0.25$$

</details>

---

### Q3. Euler 방법으로 샘플링

훈련된 모델이 벡터장 $v_\theta(x, t) = 2x + 1$을 출력한다고 하자. 초기값 $x_0 = 0$에서 시작하여 스텝 크기 $\Delta t = 0.5$로 Euler 방법을 2번 반복 적용하라:

$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t,\, t)$$

**(a)** $x_{0.5}$를 구하라.

**(b)** $x_{1.0}$을 구하라.

**(c)** 스텝 수를 늘릴수록 ($\Delta t \to 0$) 결과가 어떻게 변하는지 설명하라.

<details>
<summary>정답 보기</summary>

**(a)**

$$x_{0.5} = x_0 + 0.5 \cdot v_\theta(x_0, 0) = 0 + 0.5 \cdot (2 \cdot 0 + 1) = 0.5$$

**(b)**

$$x_{1.0} = x_{0.5} + 0.5 \cdot v_\theta(x_{0.5}, 0.5) = 0.5 + 0.5 \cdot (2 \cdot 0.5 + 1) = 0.5 + 1.0 = 1.5$$

**(c)** $\Delta t \to 0$이 되면 Euler 방법이 ODE $\dot{x} = v_\theta(x, t)$의 정확한 해에 수렴한다. 스텝이 클수록 각 구간에서 벡터장이 변함에도 불구하고 처음 값으로 직선 이동하는 오차가 누적된다. 따라서 스텝 수가 많을수록(더 촘촘하게 적분할수록) 샘플 품질이 좋아진다.

</details>

---

### Q4. 연속 방정식 (Continuity Equation)

확률 밀도 $p_t$는 벡터장 $v_t$에 의해 다음 연속 방정식을 만족해야 한다:

$$\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t(x)\, v_t(x)) = 0$$

1차원에서 $p_0 = \mathcal{N}(0, 1)$, $p_1 = \mathcal{N}(\mu, 1)$ ($\mu \neq 0$)이라 하자.

**(a)** 선형 경로 $x_t = (1-t)x_0 + t\mu$에 대응하는 벡터장 $v_t(x)$를 구하라.

**(b)** 이 $v_t$가 연속 방정식을 만족함을 확인하라. (Hint: $p_t = \mathcal{N}(t\mu,\,(1-t)^2)$)

<details>
<summary>정답 보기</summary>

**(a)** $x_t = (1-t)x_0 + t\mu$이므로 $x_0 = \dfrac{x_t - t\mu}{1-t}$. 조건부 벡터장을 $t$에 대해 미분하면:

$$v_t(x) = \frac{d}{dt}x_t = \mu - x_0 = \mu - \frac{x - t\mu}{1-t} = \frac{\mu - x}{1-t}$$

**(b)** $p_t(x) = \mathcal{N}(t\mu,\,(1-t)^2)$로 쓰면:

$$\frac{\partial p_t}{\partial t} = p_t(x) \cdot \frac{x - t\mu}{(1-t)^2}\,\mu - \frac{p_t(x)}{-(1-t)}\cdot\frac{(x-t\mu)^2-(1-t)^2}{(1-t)^2}$$

계산 과정을 요약하면:

$$\frac{\partial p_t}{\partial x}(p_t \cdot v_t) = \frac{1}{1-t}\frac{\partial}{\partial x}\left[(\mu - x)\,p_t\right] = \frac{1}{1-t}\left[-p_t + (\mu-x)\frac{\partial p_t}{\partial x}\right]$$

$\dfrac{\partial p_t}{\partial x} = -\dfrac{x - t\mu}{(1-t)^2}p_t$를 대입하면 두 항이 정확히 상쇄되어 연속 방정식 $= 0$이 성립한다.

</details>

---

### Q5. 주변 벡터장 (Marginal Vector Field)

Flow Matching의 핵심 이론적 결과는 다음이다:

$$v_t(x) = \mathbb{E}\!\left[u_t(x \mid x_1) \;\middle|\; x_t = x\right]$$

**(a)** 위 등식의 의미를 직관적으로 설명하라. (조건부 기댓값이 왜 "평균적으로 올바른" 방향을 가리키는가?)

**(b)** CFM 손실 $\mathcal{L}_{CFM}$를 최소화하는 것이 위 주변 벡터장 $v_t$를 학습하는 것과 동치임을 설명하라.

<details>
<summary>정답 보기</summary>

**(a)** 시각 $t$에 같은 위치 $x$를 지나는 궤적은 여러 개의 서로 다른 목적지 $x_1$에서 비롯될 수 있다. 각 궤적은 자신의 조건부 벡터장 $u_t(x \mid x_1)$를 갖는데, 주변 벡터장 $v_t(x)$는 "현재 $x_t = x$를 지나고 있다"는 조건 아래 이 방향들의 가중 평균이다. 이 평균 방향으로 이동하면 **전체 확률 분포를 올바르게 수송**할 수 있다.

**(b)** MSE 손실 $\mathbb{E}\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2$를 $v_\theta$에 대해 최소화하면, 각 $x_t$ 지점에서의 최적 예측은 조건부 기댓값 $\mathbb{E}[u_t \mid x_t]$이다. 이것이 바로 주변 벡터장 $v_t(x_t)$이므로, CFM 손실을 최소화하는 $v_\theta$는 $v_t$를 학습한다.

</details>

---

### Q6. Diffusion vs Flow Matching 비교

Diffusion 모델의 forward process는 다음과 같다:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

반면 Flow Matching의 경로는 $x_t = (1-t)x_0 + t x_1$이다.

| 항목                 | Diffusion                           | Flow Matching                |
| -------------------- | ----------------------------------- | ---------------------------- |
| 경로 형태            | 비선형 (noise schedule)             | 선형 보간                    |
| 목표 함수            | $\epsilon$-prediction (노이즈 예측) | $v$-prediction (벡터장 예측) |
| 추론 step 수         | 수십~수백                           | 수십 이하 가능               |
| ODE Solver 필요 여부 | ?                                   | ?                            |

**(a)** 표의 빈칸을 채워라.

**(b)** Flow Matching이 Diffusion보다 적은 추론 step으로도 좋은 성능을 낼 수 있는 이유를 경로의 기하학적 관점에서 설명하라.

<details>
<summary>정답 보기</summary>

**(a)**

| 항목                 | Diffusion                           | Flow Matching                |
| -------------------- | ----------------------------------- | ---------------------------- |
| 경로 형태            | 비선형 (noise schedule)             | 선형 보간                    |
| 목표 함수            | $\epsilon$-prediction (노이즈 예측) | $v$-prediction (벡터장 예측) |
| 추론 step 수         | 수십~수백                           | 수십 이하 가능               |
| ODE Solver 필요 여부 | 필요 (DDIM 등 ODE sampler)          | 필요 (Euler / RK4 등)        |

**(b)** Diffusion은 noise schedule로 인해 궤적이 **곡선**을 그리므로, 이를 충실하게 수치 적분하려면 많은 스텝이 필요하다. 반면 Flow Matching의 선형 보간 경로는 **직선**이다. 직선 궤적은 Euler 방법 같은 1차 적분기로도 큰 오차 없이 따라갈 수 있으므로, 적은 함수 평가(NFE, Number of Function Evaluations)로 동등한 품질의 샘플을 생성할 수 있다.

</details>
