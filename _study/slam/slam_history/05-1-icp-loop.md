---
layout: study-chapter
title: "5.1 The ICP loop"
description: "Aligning two scans by alternating matching and solving."
importance: 5.1
category: SLAM
series: slam_history
permalink: /study/slam/history/05-1-icp-loop/
---

> **Goal:** run ICP by hand in 1D and explain the chicken-and-egg problem it solves.

The robot scans a corner, drives a bit, and scans again. Both scans show the same corner in two different sensor frames. **The transform that slides one onto the other is the robot's motion.**

---

### Problem 1 — known partners

No rotation, one axis. Source points `[0, 1, 2]` match target points `[2, 3, 4]` in order. What shift $t$ aligns them?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

What shift does each pair want?

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Every pair wants $+2$, so $t = 2$ with zero residual. With correspondences known, registration is easy — it is just an average.

</details>

### Problem 2 — the objective

Point-to-point ICP minimises $\sum_i \|Rp_i + t - q_{c(i)}\|^2$. Say in words what the term inside the norm is.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

“Where the transform puts my point, minus where its partner actually is” — the leftover gap. ICP squares and sums the gaps and picks the $R, t$ that make the total smallest.

</details>

### Problem 3 — chicken and egg

In a real scan nobody tells you which target point is each source point's partner. What do you need to find partners? What do you need to find the transform?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

To pick partners you need the transform; to find the transform you need the partners. **ICP alternates:**

```text
  guess a transform (often: robot barely moved)
    ↓
  ① each source point's partner = nearest target point
  ② hold partners fixed, solve for the best transform
    ↓
  repeat ① with the improved transform
```

</details>

### Problem 4 — one bad partner

Same as Problem 1, but the third source point is wrongly matched to a target at 40. What shift does least squares give now?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Average of what each pair wants.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Pairs want $+2, +2, +38$, so $t = 42/3 = 14$. **One bad match overrules two perfect ones.**

</details>

### Problem 5 — why so fragile?

At $t = 14$, the good pairs are 12 off and the bad pair is 24 off. Why does the optimiser accept ruining the two good fits?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Squaring. Being 38 off at $t = 2$ costs $38^2 = 1444$; at $t = 14$ the cost is $2 \times 144 + 576 = 864$. The optimiser happily trades. The **median** of $(2, 2, 38)$ is 2 — robust losses and outlier rejection exist to behave more like that.

</details>

### Problem 6 — challenge — the point-to-plane Jacobian

With residual $r_i = n_i^{\top}(Rp_i + t - q_i)$ and a small left rotation $R' \approx (I + [\delta\theta]_\times)R$, find the Jacobian row with respect to $[\delta\theta,\ \delta t]$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Let $a_i = Rp_i$. Use $[\delta\theta]_\times a = -[a]_\times\delta\theta$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$r_i' \approx r_i + n_i^{\top}(-[a_i]_\times\delta\theta + \delta t)$, so $J_i = \begin{bmatrix}-n_i^{\top}[a_i]_\times & n_i^{\top}\end{bmatrix}$. With a right perturbation the rotation part changes — match the convention to the code.

</details>

---

**You can now:** run ICP's alternation by hand, compute a least-squares shift, and show how one wrong correspondence ruins it.
