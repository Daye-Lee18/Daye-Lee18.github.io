---
layout: post
title: 강화학습 깊이있게 이해하기 - HER과 Exploration-Exploitation Trade-off
date: 2026-05-21
category: [Machine Learning, Reinforcement Learning]
tags: [강화학습, HER, Exploration, Exploitation, 정책학습]
author: dayelee
---

회사 프로젝트에서 접한 HER(Hindsight Experience Replay)과 exploration-exploitation의 균형을 깊이있게 정리한 학습 노트입니다.

## 1. 강화학습의 기초

### 강화학습의 정의

강화학습(RL)은 에이전트가 환경과의 상호작용을 통해 보상 신호를 최대화하는 정책을 학습하는 패러다임입니다.

```
Agent → Action → Environment → State, Reward → Agent
```

핵심 요소:

- **State (상태)**: 현재의 관찰 정보
- **Action (행동)**: 에이전트가 취할 수 있는 선택지
- **Reward (보상)**: 환경이 주는 피드백
- **Policy (정책)**: State → Action의 매핑

---

## 2. Exploration vs Exploitation: 핵심 딜레마

### 문제 정의

강화학습에서 가장 중요한 균형:

- **Exploitation (활용)**: 현재까지 알려진 최고의 전략을 사용

  - "내가 이미 알고 있는 가장 좋은 방법을 계속 쓰자"
  - 즉시 보상 극대화
  - 새로운 전략 발견 불가능

- **Exploration (탐색)**: 미지의 행동들을 시도
  - "혹시 더 좋은 방법이 있지 않을까?"
  - 장기적으로 최적의 정책 발견
  - 단기 보상은 감소할 수 있음

### 왜 중요한가?

**예시: 로봇 팔 제어**

```
초기 상태: 로봇이 목표 위치에 도달하는 방법을 모름

Pure Exploitation:
→ 첫 시도에 우연히 팔을 조금 움직임 (작은 보상)
→ 계속 같은 방식만 반복 (최적해 못 찾음)

Pure Exploration:
→ 무작위로 계속 시도 (비효율적)
→ 언젠가는 목표에 도달하지만 매우 오래 걸림

Balanced Approach:
→ 효과적인 행동은 자주, 새로운 행동도 종종 시도
→ 빠르게 최적의 전략 학습 가능
```

### 일반적인 Exploration 전략

1. **ε-Greedy**: 확률 ε로 탐색, (1-ε)로 활용
2. **Softmax**: 모든 행동에 확률을 할당 (좋은 행동에 높은 확률)
3. **Upper Confidence Bound (UCB)**: 불확실성이 높은 행동 선호
4. **Entropy Regularization**: 정책의 entropy를 보상에 추가

---

## 3. Hindsight Experience Replay (HER) 상세 분석

### 왜 HER이 필요한가?

**문제: Sparse Reward 환경**
많은 실제 문제는 목표에 도달할 때만 보상을 줍니다.

```
Episode:
Step 1: Action → Reward = 0
Step 2: Action → Reward = 0
Step 3: Action → Reward = 0
...
Step 50: Action → GOAL REACHED! → Reward = 1

문제: 49번의 실패에서 배울 정보가 거의 없음!
```

### HER의 핵심 아이디어

**"실패도 성공으로 만들 수 있다면?"**

HER은 실패한 에피소드를 다시 해석합니다:

```
원래 목표: Position A에 도달하기

Episode:
Step 1: Start → Position P1 (목표 A에 가까워짐, 하지만 실패)
Step 2: P1 → Position P2 (더 가까워짐, 하지만 실패)
...
Step 50: P49 → Position P50 (목표 A 못 도달)

HER의 재해석:
"Step 50에서 도달한 Position P50을 새로운 목표로 설정하면?"
→ "P50에 도달하는 것"은 이미 성공한 에피소드!

이제 이 에피소드로부터:
- 어떻게 P50에 도달할 수 있는가를 학습
- 미래에 비슷한 상태에서 도움이 될 경험
```

### HER의 알고리즘

```python
# 에피소드 수집
for episode in range(num_episodes):
    trajectory = collect_trajectory(policy, goal)

    # 일반적인 학습: 원래 목표로 학습
    rewards = compute_reward(trajectory, goal)
    update_policy(rewards)

    # HER: 다른 목표들로도 학습
    for hindsight_goal in sample_goals_from_trajectory(trajectory):
        hindsight_rewards = compute_reward(trajectory, hindsight_goal)
        update_policy(hindsight_rewards)
```

### HER의 장점

1. **Sample Efficiency**: 한 번의 에피소드에서 여러 번 학습
2. **Sparse Reward 극복**: 실패한 경험도 재활용
3. **다양한 목표 학습**: 같은 궤적으로 여러 목표 도달 능력 학습
4. **비용 효율적**: 실제 인터랙션 횟수 감소

### HER의 변형들

- **Final State**: 에피소드의 마지막 상태를 새 목표로
- **Random**: 같은 에피소드 내의 무작위 상태
- **Episode**: 최근 k개 에피소드에서 무작위로
- **Future**: 현재 step 이후의 상태만 선택

---

## 4. Success Rate 기반 학습

### 개념

회사 모델에서 사용하는 방식으로 보이는데, 이는 다음과 같은 특징이 있습니다:

```
전통적 접근: 목표 달성 여부가 아니라 보상 최대화
Success Rate 방식: "주어진 현재 상태에서 얼마나 자주 성공하는가"

예:
- 상태: 현재 로봇 팔의 위치 (P_current)
- 목표: 특정 위치에 도달 (P_goal)
- 학습: P_current → success rate 예측
```

### 왜 효과적인가?

1. **상태-성공률 매핑**: 각 상태에서의 달성 가능성을 학습
2. **보상 신호 강화**: Success/Failure는 명확한 신호
3. **일반화**: 특정 목표가 아닌 상태 자체의 특성 학습
4. **정책 향상**: 성공률 높은 행동 선택

### HER과의 결합

```
HER + Success Rate:
1. 여러 에피소드 수집
2. 각 에피소드에서 도달한 상태들 기록
3. 각 상태에 대해 "그 상태에서 목표 도달 확률" 계산
4. 상태 → 성공률 함수 학습
5. 정책은 성공률이 높은 상태로 이동하는 행동 선택
```

---

## 5. 실무 적용 팁

### HER 사용할 때 고려사항

1. **Reward Function 설계**

   ```python
   # 좋은 설계: 명확한 목표와 연속적인 reward
   def reward(achieved_state, goal_state):
       distance = euclidean_distance(achieved_state, goal_state)
       return 0.0 if distance < threshold else -distance
   ```

2. **Goal Sampling 전략**

   - 모든 과거 상태를 목표로 사용하면 계산 비용 증가
   - 샘플링 비율을 조정하여 균형 맞추기

3. **Exploration 설정**
   - HER로 많은 데이터를 재활용하므로, exploration은 적당히
   - ε을 너무 작게 하면 새로운 궤적 탐색 못 함

### 성공 지표

```python
# 학습 진행도 모니터링
metrics = {
    "success_rate": # 목표 달성 확률
    "exploration_rate": # 새로운 상태 발견 비율
    "sample_efficiency": # 필요한 에피소드 수 감소
    "convergence_speed": # 학습 속도
}
```

---

## 6. 정리: 핵심 개념 다시 보기

| 개념             | 설명                   | 목표                    |
| ---------------- | ---------------------- | ----------------------- |
| **Exploration**  | 미지의 행동 시도       | 전역 최적해 찾기        |
| **Exploitation** | 알려진 최고 전략 사용  | 즉시 보상 최대화        |
| **Balance**      | 둘 사이의 균형         | 효율적인 학습           |
| **HER**          | 실패를 성공으로 재해석 | Sample efficiency 증가  |
| **Success Rate** | 상태별 성공 확률 학습  | 정책 개선의 명확한 신호 |

---

## 7. 추가 학습 자료

### 핵심 논문

- Andrychowicz et al., 2017: "Hindsight Experience Replay" (NIPS)
  - HER의 원본 논문

### 관련 개념들

- **DQN** (Deep Q-Network): 기본적인 심화학습
- **Policy Gradient**: 정책을 직접 최적화
- **Actor-Critic**: Value와 Policy를 동시에 학습
- **Curiosity-driven Exploration**: 내재적 motivation으로 탐색

---

## 8. 실습 아이디어

**로봇 팔을 사용한 HER 구현**

```python
# 1단계: 환경 설정 (OpenAI Gym)
env = RoboticFetchPickAndPlace()

# 2단계: Replay Buffer with HER 구현
buffer = HERReplayBuffer(capacity=1e6)

# 3단계: Agent 학습
for episode in range(num_episodes):
    trajectory = env.rollout(policy)
    buffer.add(trajectory)

    # HER: trajectory에서 다양한 목표로 학습
    batch = buffer.sample_with_her()
    update_network(batch)

    # 평가
    success_rate = evaluate_policy(policy)
    print(f"Episode {episode}: Success Rate = {success_rate:.2%}")
```

---

**다음 학습 단계:**

1. HER 원본 논문 정독
2. 간단한 환경(reaching task)에서 HER 구현 및 테스트
3. Exploration 전략 비교 실험
4. Success rate 기반 정책 최적화 구현

이 개념들이 회사 프로젝트에서 어떻게 적용되었는지 이해하면, 더 깊이있는 모델 개선이 가능할 것입니다!
