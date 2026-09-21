---
layout: study-chapter
title: "6.1 Constraints and least squares"
description: "A pose graph solved by hand."
importance: 6.1
category: SLAM
series: slam_history
permalink: /study/slam/history/06-1-constraints/
---

> **Goal:** set up a tiny pose graph, solve it, and explain how a loop closure spreads error.

---

### Problem 1 — what a filter forgets

A filter keeps only the current pose $x_3$. A graph keeps $x_0, x_1, x_2, x_3$. New information arrives about $x_1$. Which one can use it?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Only the graph. The filter folded $x_1$ into its estimate and there is no $x_1$ left to correct. **In Graph SLAM every pose stays an editable variable.**

</details>

### Problem 2 — measurements become constraints

Odometry says $x_1 - x_0 = 1$. Write the residual (how badly the constraint is violated).

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$r = (x_1 - x_0) - 1$. Zero when the poses agree with the measurement. Graph SLAM picks poses that make the **weighted sum of squared residuals** smallest:
$$X^* = \arg\min_X \sum_k r_k^{\top}\Omega_k r_k$$
$\Omega_k$ is the inverse covariance — a confident measurement pulls harder.

</details>

### Problem 3 — solve a tiny graph

Fix $x_0 = 0$. Odometry: $x_1 - x_0 = 1$, $x_2 - x_1 = 1$. Loop: $x_2 - x_0 = 1.8$. Equal weights. Find $x_1, x_2$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Minimise $E = (x_1 - 1)^2 + (x_2 - x_1 - 1)^2 + (x_2 - 1.8)^2$ by setting both partial derivatives to zero.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$2x_1 - x_2 = 0$ and $-x_1 + 2x_2 = 2.8$, so $x_1 \approx 0.933$, $x_2 \approx 1.867$.
The 0.2 m disagreement is shared: each odometry step is shortened, and the loop is not met exactly either.

</details>

### Problem 4 — naive fix vs graph

Three odometry steps of 1 m; a loop says $x_3$ is 2.7 m from $x_0$. Compare the total squared violation of (a) moving only $x_3$ to 2.7, and (b) the least-squares answer $x = (0.925, 1.85, 2.775)$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
            poses                     residuals                 Σr²
 (a)  1.0, 2.0, 2.7        odom 0, 0, −0.3 ; loop 0            0.090
 (b)  0.925, 1.85, 2.775   odom −0.075 ×3 ; loop +0.075        0.0225
```

(b) is four times better. It spreads the error evenly and **does not even satisfy the loop exactly** — the loop is one more noisy measurement, not a decree.

</details>

### Problem 5 — weight the loop

In Problem 4, give the loop 100× the weight of each odometry step. Which way does the solution move?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Towards 2.7: roughly $x = (0.900, 1.801, 2.701)$. A larger $\Omega$ means more trust, so the loop is honoured almost exactly and the odometry absorbs the correction.

</details>

### Problem 6 — challenge — gauge freedom

A graph containing only relative measurements has the same cost if you shift every pose by +5 m. Why, and what do you do about it?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Every residual is a difference of poses, so a global shift cancels. The solution is not unique. **Fix one pose** (or add a prior) to anchor the graph.

</details>

---

**You can now:** turn measurements into residuals, solve a small pose graph, and explain why least squares spreads loop error instead of dumping it.
