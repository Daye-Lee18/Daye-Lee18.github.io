---
layout: study-chapter
title: "7.1 Why the matrix is mostly zeros"
description: "Sparsity makes large graphs solvable."
importance: 7.1
category: SLAM
series: slam_history
permalink: /study/slam/history/07-1-mostly-zeros/
---

> **Goal:** explain why graph SLAM's linear system is sparse and why that makes it tractable.

Uses Chapter 6.

---

### Problem 1 — the naive cost

Chapter 6's least squares with 10,000 poses has 10,000 unknowns. Inverting an $n \times n$ matrix costs about $n^3$. How many operations is that?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$10^{12}$ — hopeless to do every step. Yet large graphs are solved in real time. Something about the matrix must help.

</details>

### Problem 2 — one row

The odometry measurement between $x_5$ and $x_6$ becomes one row of the Jacobian. Which variables does that row touch?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Only $x_5$ and $x_6$. It says nothing about $x_{900}$, so every other entry of that row is **zero**.

</details>

### Problem 3 — how empty?

If every row touches 2 of 10,000 variables, what fraction of each row is zero?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$1 - 2/10000 = 99.98\%$. The matrix is almost entirely zeros — it is **sparse**. Solvers use QR or Cholesky factorisations that skip the zeros instead of forming an inverse.

</details>

### Problem 4 — challenge — incremental is not constant

iSAM2 reuses the previous factorisation when a new factor arrives. Why can it still sometimes be slow?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A loop closure linking far-apart poses touches a large part of the Bayes tree, so much of it must be refactorised; big state changes also trigger relinearisation. **Incremental means “reuse what you can”, not “every update is O(1)”.**

</details>

---

**You can now:** explain why each measurement makes a sparse row and why sparsity, not raw computing power, is what makes large graph SLAM tractable.
