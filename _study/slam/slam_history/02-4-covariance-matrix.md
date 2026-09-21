---
layout: study-chapter
title: "2.4 The covariance matrix"
description: "Fill in the EKF-SLAM covariance and watch one observation correct an unobserved landmark."
importance: 2.4
category: SLAM
series: slam_history
permalink: /study/slam/history/02-4-covariance-matrix/
---

> **Goal:** know what is in the EKF-SLAM state, fill in its covariance, and use it.

Uses Parts 2.1 and 2.2.

---

### Problem 1 — what is in the state?

Which does EKF-SLAM keep?

```text
(a)  [ x_0, x_1, …, x_t, m_1, …, m_n ]     every pose ever, plus landmarks
(b)  [ x_t, m_1, …, m_n ]                  current pose only, plus landmarks
```

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

**(b).** When the robot moves, the old pose is replaced, not appended. Its information survives only as correlations in the covariance.
(a) is Graph SLAM (Chapter 6), where old poses stay editable.

</details>

### Problem 2 — size

A 2D robot ($x, y, \theta$) with 3 landmarks ($x, y$ each). How long is the state? How big is the covariance?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$3 + 2 \times 3 = 9$, so $9 \times 9 = 81$ entries.

</details>

### Problem 3 — the wall

Same, with 1000 landmarks.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$2003 \times 2003 \approx 4$ million entries, and every update touches all of them. This $O(n^2)$ cost is what FastSLAM attacks in Chapter 3.

</details>

### Problem 4 — one landmark's variance

1D again. Robot variance 0.25. Range noise 0.01. Landmark A is placed at robot + range. What is $\mathrm{Var}(A)$?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Independent errors add their variances.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$0.25 + 0.01 = 0.26$. A is almost entirely as uncertain as the robot.

</details>

### Problem 5 — the shared part

Using Part 2.2: what is $\mathrm{Cov}(A, B)$ for two landmarks seen from the same pose? And $\mathrm{Cov}(\text{robot}, A)$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Both equal the robot's variance, **0.25** — the shared error from Part 2.2.

</details>

### Problem 6 — fill it in

Write the $3 \times 3$ covariance for $[\text{robot}, A, B]$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
            robot    A       B
   robot    0.25    0.25    0.25
   A        0.25    0.26    0.25
   B        0.25    0.25    0.26
```

**Every off-diagonal = 0.25 = the robot's own variance.** The number the landmarks share is literally the pose error.

</details>

### Problem 7 — watch it work

Re-measure only A, very precisely ($z = 8.2$, noise $10^{-4}$), from estimates $[5.0,\ 8.0,\ 12.0]$.
Without computing, predict what happens to B. Then check against the result below.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

B should move by about the same as A, because of the 0.25 linking them. One EKF update gives:

```text
   gain K = [0.961, 1.000, 0.961]      ← B has a large gain but was never observed

               before    after    truth
   robot        5.00     5.192     5.2
   A            8.00     8.200     8.2
   B           12.00    12.192    12.2
```

This is Part 2.2 Problem 4 done by the actual filter. The correction travelled through the off-diagonal.

</details>

---

**You can now:** say what EKF-SLAM keeps, size its covariance, fill it in from first principles, and explain how one observation corrects an unobserved landmark.

**Next:** [Chapter 3 — FastSLAM]({{ '/study/slam/history/03-fastslam/' | relative_url }}) removes the $O(n^2)$ matrix from Problem 3.
