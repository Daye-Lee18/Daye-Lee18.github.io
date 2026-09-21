---
layout: study-chapter
title: "2.2 Errors spread"
description: "Why landmarks seen from the same pose share an error."
importance: 2.2
category: SLAM
series: slam_history
permalink: /study/slam/history/02-2-errors-spread/
---

> **Goal:** explain why landmarks become correlated, and why that is useful.

---

### Problem 1

The robot **believes** it is at $x = 5.0$ but is **really** at $x = 5.2$.
It measures pillar A exactly 3.0 m ahead. Where does the robot think A is? Where is A really?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Thinks: $5.0 + 3.0 = 8.0$. Really: $5.2 + 3.0 = 8.2$.

The range was measured perfectly, yet A is placed 0.2 m wrong — it inherited the pose error.

</details>

### Problem 2

From the same spot it measures pillar B 7.0 m ahead. Thinks? Really?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Thinks 12.0, really 12.2.

</details>

### Problem 3

Compare the two errors. What do they have in common, and why?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

**Both are +0.2, same size, same direction** — both came from the robot's one mistaken pose.
Two measurements that were independent produced two landmark errors that are **not** independent.
That shared part is what **correlation** means here.

</details>

### Problem 4

Later a trustworthy source says A is really at 8.2. How wrong was the pose? Where is B, even though you never looked at B again?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A was 0.2 short, so the pose was 0.2 short, so **B is at 12.2**. One observation corrected two things you did not observe.

</details>

### Problem 5

What must a filter **remember** to be able to do Problem 4?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

That A, B and the pose share an error — not just each one's own uncertainty.
That memory lives in the **off-diagonal entries of the covariance matrix** (Part 2.4).
Throw them away and Problem 4 becomes impossible.

</details>

### Problem 6 — challenge

A teammate zeroes all robot–landmark cross-covariances "to save computation". What goes wrong?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Correlated landmarks get treated as independent evidence. The filter counts one pose error many times over,
becomes **overconfident**, and stops propagating corrections like Problem 4.
Approximations are allowed, but you must say what you dropped and how you guard consistency.

</details>

---

**You can now:** say why landmarks measured from one pose are correlated, and why that correlation is what lets one observation fix many landmarks.
