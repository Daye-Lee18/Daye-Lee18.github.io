---
layout: study-chapter
title: "3.2 Particles, weights and resampling"
description: "How FastSLAM picks among path guesses."
importance: 3.2
category: SLAM
series: slam_history
permalink: /study/slam/history/03-2-particles/
---

> **Goal:** score particles against a measurement, compute effective sample size, and explain when particles beat a single Gaussian.

Uses Part 3.1.

---

### Problem 1 — what a particle holds

True or false: a particle is one guessed robot position.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

False. **Each particle carries a whole path guess and its own private map** built on that guess. Three particles means three different maps.

</details>

### Problem 2 — score them

Three particles predict the door at 9.8, 10.0 and 10.3 m. The sensor measures 10.0 m. Rank their weights.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
 particle   predicted   miss   weight
    1          9.8       0.2    low
    2         10.0       0.0    high
    3         10.3       0.3    lowest
```

The path was never solved directly — it is decided by **which guess keeps predicting well.**

</details>

### Problem 3 — resampling

What does resampling do with those weights, and what happens to particle 3?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It copies high-weight particles and drops low-weight ones, so the next round is mostly paths like particle 2. Particle 3 is probably discarded.

</details>

### Problem 4 — effective sample size

Weights $[0.1,\ 0.3,\ 0.6]$ (already sum to 1). Compute $N_{eff} = 1/\sum w_i^2$. With a resampling threshold of $N/2 = 1.5$, do you resample?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Square each weight, add, invert.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$N_{eff} = 1/0.46 \approx 2.17 > 1.5$, so **no**.
For intuition: equal weights give $N_{eff} = 3$ (all useful); $[0.98, 0.01, 0.01]$ gives $\approx 1.04$ (effectively one particle).

</details>

### Problem 5 — two identical doors

Two identical doors at $x = 10$ and $x = 30$; the robot is at one of them, unsure which. What single mean would an EKF report? What would particles report?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

EKF: one mean, **20** — the one place the robot certainly is not.
Particles: something like `[10, 10, 10, 30, 30, 30]` — **two live answers, nothing in between.** A single Gaussian cannot express that.

</details>

### Problem 6 — challenge — resampling too often

Why can resampling too eagerly be harmful?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It collapses diversity — copies of a few particles crowd out alternatives (**particle impoverishment**). If the survivors were wrong, no correct hypothesis remains to recover to. Resampling too late wastes compute on hopeless particles, so $N_{eff}$ thresholds balance the two.

</details>

---

**You can now:** score and resample particles, compute $N_{eff}$, and explain why particles can represent “here or there” when an EKF cannot.
