---
layout: study-chapter
title: "1.1 Relative measurements and drift"
description: "Why odometry drifts, by problems."
importance: 1.1
category: SLAM
series: slam_history
permalink: /study/slam/history/01-1-relative-measurements/
---

> **Goal:** explain what odometry actually measures and why its error only grows.

---

### Problem 1 — no maths

A robot's sensor reports “there is a wall 3 m ahead of me.” Can the robot tell where it is in the building from that alone?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No. Any place with a wall 3 m in front is consistent with the reading — room A, corridor B, room C.
And the reverse also fails: to draw that wall on a map you need to know where the robot stood.
**Each of the two problems needs the other's answer first.** Estimating both together is SLAM.

</details>

### Problem 2 — what odometry reports

Odometry reports once per second. The robot starts at $x = 0$ and odometry says:

```text
t = 0 → 1    +1.00 m
t = 1 → 2    +1.00 m
t = 2 → 3    +1.00 m
```

Where is the robot at $t = 3$? Did the sensor ever tell you that number directly?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$x = 3.00$. **The sensor never said “3.00”** — it only ever said “+1.00”.
The absolute position is something _we_ computed by adding relative measurements, starting from a position we assumed.
That is the defining property of odometry: **relative to the previous pose, never absolute.**

</details>

### Problem 3 — a tiny error

The wheels slip a little: the robot really moves 1.00 m per step but odometry reports 1.02 m. What is the position error after 1, 10 and 100 steps?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Each step adds 0.02 of error, and nothing ever removes it.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
 step   odometry sum   true    error
   1        1.02        1.00    0.02
  10       10.20       10.00    0.20
 100      102.00      100.00    2.00
```

A 2 cm error you would never notice becomes 2 m. **Nothing went wrong at step 100 — the error was accumulated, not created.**

</details>

### Problem 4 — heading is worse

Every distance is measured perfectly, but the initial heading is wrong by 1°. How far sideways is the robot after driving 100 m straight?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Sideways offset ≈ distance × sin(angle).

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$100 \times \sin 1° \approx 1.75$ m. Heading error rotates _every_ step that follows, so it grows with distance even with perfect wheels.

</details>

### Problem 5 — two independent errors

On one axis, two moves each carry an independent error with standard deviation 0.1 m. What is the standard deviation of their combined error?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Independent errors add their **variances**, not their standard deviations.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\sqrt{0.1^2 + 0.1^2} \approx 0.141$ m — not 0.2. Variances add; standard deviations add in quadrature.
(Chapter 2.1 builds on exactly this.)

</details>

### Problem 6 — closing a square

The robot drives a 10 m square and stops exactly where it started. Its summed odometry says it ended at $(0.4, 0.3)$. How big is the gap, and what does the robot's own camera say?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The gap is $\sqrt{0.4^2 + 0.3^2} = 0.5$ m. The camera sees the front door it started at — **it knows it is back, its arithmetic disagrees.** That 0.5 m is the drift the sum collected. Part 1.2 is about using that disagreement.

</details>

---

**You can now:** say what odometry measures, why its error only grows, and how heading error differs from distance error.
