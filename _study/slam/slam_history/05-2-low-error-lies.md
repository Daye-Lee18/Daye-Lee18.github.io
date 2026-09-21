---
layout: study-chapter
title: "5.2 When a low error lies"
description: "Converged, tiny residual, wrong pose."
importance: 5.2
category: SLAM
series: slam_history
permalink: /study/slam/history/05-2-low-error-lies/
---

> **Goal:** recognise the two ways ICP can converge confidently to a wrong answer.

Uses Part 5.1.

---

### Problem 1 — repeated pillars

A corridor has pillars every 5 m. The robot really moved 5 m. ICP matches each pillar to the _next_ one instead of itself. What residual and what motion does ICP report?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Residual ≈ 0, motion ≈ 0 m. Every pair lines up to within millimetres, and the answer is off by a full 5 m. **The optimiser did its job; the correspondences lied.** That is why the initial guess and overlap matter.

</details>

### Problem 2 — one flat wall

Point-to-plane measures distance along the wall's normal. The robot sees only one flat wall and slides 1 m along it. How do the residuals change?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

They do not change at all. Sliding along the wall leaves every point-to-plane distance the same, so **that direction is unobservable** — not badly estimated, simply absent from the data.

</details>

### Problem 3 — how would you notice?

In Problem 2, what property of the least-squares normal matrix $H = J^{\top}J$ reveals the unobservable direction?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A (near-)zero **eigenvalue**, with its eigenvector pointing along the wall. A large condition number is the warning sign. Chapter 2.3 Problem 6 was the same story for a single range.

</details>

### Problem 4 — challenge — three causes

ICP converged with a low residual but the pose is wrong. Give three causes and a diagnostic for each.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

1. **Local minimum** in repetitive structure → change the initial guess, compare with a global registration.
2. **Bad correspondences / outliers** → check distance and normal-angle distributions, inlier ratio.
3. **Degeneracy** (plane, long corridor) → eigenvalues / condition number of $J^{\top}J$.
   Also: low overlap or scan distortion — look at raw vs deskewed scans.

</details>

---

**You can now:** explain why a converged ICP with a tiny residual can still be wrong, and how to detect repeated-structure and degeneracy failures.
