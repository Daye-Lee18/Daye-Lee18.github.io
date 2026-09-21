---
layout: study-chapter
title: "2.3 Linearisation"
description: "Replace a curve by its tangent — and see when that fails."
importance: 2.3
category: SLAM
series: slam_history
permalink: /study/slam/history/02-3-linearisation/
---

> **Goal:** say what "linearise" means, compute a Jacobian, and predict when it breaks.

Part 2.1's formula only works when the measurement is a straight-line function of the state.
Real sensors are curves. The fix is to pretend the curve is straight near where you are.

---

### Problem 1 — warm-up

$y = x^2$. What is $y$ at $x = 3$? At $x = 3.1$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

9 and 9.61.

</details>

### Problem 2 — the straight line

The slope of $x^2$ at $x = 3$ is $2x = 6$. Using only that slope, estimate $y$ at $x = 3.1$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Start at 9, go 0.1 to the right, rise 6 per unit.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$9 + 6 \times 0.1 = 9.6$. True value 9.61, error 0.01.

That is linearisation: **value here + slope × step**. Formally, $h(x) \approx h(\hat x) + H(x - \hat x)$ with $H$ the derivative.

</details>

### Problem 3 — going further

Same straight line, now at $x = 3.5$ and $x = 4$. Errors?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
   x     true    line    error
  3.1    9.61    9.60    0.01
  3.5   12.25   12.00    0.25
  4.0   16.00   15.00    1.00
```

Step ×5 → error ×25. **Error grows with the square of the step.**
Excellent nearby, useless far away — and "far" is measured in units of your own uncertainty.

</details>

### Problem 4 — a real sensor

Robot at $(0,0)$, landmark at $(3,4)$. The sensor measures range $h = \sqrt{(3-x)^2 + (4-y)^2}$. What does it read?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

5. That square root is not a matrix multiplication — Part 2.1's algebra cannot use it directly.

</details>

### Problem 5 — intuition first

The robot steps 0.1 m **straight towards** the landmark. How much does the range change?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It drops by exactly 0.1 m. No calculus needed.

</details>

### Problem 6

Now it steps 0.1 m **sideways** — perpendicular to the line to the landmark. Range change?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Almost zero (true value +0.001). Moving around a circle centred on the landmark keeps the range the same.

**One range measurement says nothing about the sideways direction.** That is why a single range cannot fix a 2D position.

</details>

### Problem 7 — the Jacobian

Problems 5–6 describe the slope of $h$ in two directions. The unit vector towards the landmark is $(0.6, 0.8)$.
Guess $H = \partial h / \partial (x, y)$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Moving along $(0.6, 0.8)$ should change $h$ by $-1$ per metre; moving perpendicular, by 0.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$H = [-0.6,\; -0.8]$ — minus the unit vector towards the landmark. Check: $H \cdot (0.06, 0.08) = -0.1$ ✓, and $H \cdot (-0.08, 0.06) = 0$ ✓.

That is the linearisation of the range sensor. It is a matrix, so Part 2.1's machinery works again.
**Linearising is taking the derivative; what comes out is the matrix $H$.**

</details>

### Problem 8 — challenge

Using Problem 3, if your position uncertainty doubles, roughly how much worse does the linearisation get? What does that suggest about starting an EKF with a poor initial guess?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

About **4×** (error ∝ step²). A poor guess means bad Jacobians, which means a worse estimate, which means worse Jacobians — the feedback loop behind **EKF divergence**. A good initial estimate is worth far more than it looks.

</details>

---

**You can now:** linearise a function around a point, derive the range Jacobian from geometry, and explain why EKFs need good initial guesses.
