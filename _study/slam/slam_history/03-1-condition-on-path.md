---
layout: study-chapter
title: "3.1 Why condition on the path"
description: "The trick that removes EKF-SLAM's giant matrix."
importance: 3.1
category: SLAM
series: slam_history
permalink: /study/slam/history/03-1-condition-on-path/
---

> **Goal:** explain what “landmarks are independent given the path” means and why it matters.

Uses Chapter 2.2 and 2.4.

---

### Problem 1 — the cost

Chapter 2.4 showed a 2D EKF with 1000 landmarks has a $2003 \times 2003$ covariance, updated in full every step. Why did every landmark have to be linked to every other?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Because they were all measured from the same **uncertain** robot pose (Chapter 2.2). Shared pose error made them correlated, and correlations have to be stored.

</details>

### Problem 2 — a hypothetical

Suppose someone told you the robot's **exact** path. Would pillars A and B from Chapter 2.2 still be correlated?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Where did their shared error come from?

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No. With the pose known exactly, A's estimate depends only on A's own measurement and B's only on B's. **Nothing is shared any more.**

</details>

### Problem 3 — what that buys

If the path were known, how would you store 1000 landmarks instead of one $2003 \times 2003$ matrix?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

As 1000 tiny independent $2 \times 2$ problems. That is the core of **FastSLAM (2002)**: once the path is given, the landmark estimates decouple.

</details>

### Problem 4 — but we don't know the path

Nobody hands us the true path. How can FastSLAM still use the trick?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It **guesses** the path many times over. Each guess is a **particle**, and inside each particle the path is treated as known — so the cheap decoupled version is legal there. Part 3.2 is about scoring those guesses.

</details>

### Problem 5 — challenge — the exact condition

Why is it wrong to say “in FastSLAM the map points are independent of each other”?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

They are independent only **conditionally on the path** (plus model assumptions like static landmarks and known data association). Marginally — with the path uncertain — the same path error still affects several landmarks, so they depend on each other.

</details>

---

**You can now:** state the conditional-independence trick and why it replaces one huge matrix with many small ones.
