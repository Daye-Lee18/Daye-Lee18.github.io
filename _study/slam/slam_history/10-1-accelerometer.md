---
layout: study-chapter
title: "10.1 What an accelerometer measures"
description: "Specific force, gravity, bias — and why integration explodes."
importance: 10.1
category: SLAM
series: slam_history
permalink: /study/slam/history/10-1-accelerometer/
---

> **Goal:** explain specific force, compute bias and attitude-error drift, and say why IMUs need aiding.

---

### Problem 1 — a still robot

A robot sits motionless on a table. What does its accelerometer read?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

About $(0, 0, +9.81)$ m/s², not zero. An accelerometer measures **specific force**, which includes the reaction to gravity. It is not the robot's acceleration.

</details>

### Problem 2 — integrate the raw reading

Integrate that reading twice for 10 s as if it were acceleration. How far has the robot “moved”?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

$\frac12 a t^2$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\frac12 \times 9.81 \times 10^2 \approx 490$ m — straight up. Gravity must be subtracted first, which requires knowing which way is down, i.e. the orientation.

</details>

### Problem 3 — a tiny bias

After removing gravity, a constant bias of 0.01 m/s² remains (one thousandth of gravity). Velocity and position error after 1, 10 and 60 s?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

$\delta v = bt$, $\delta p = \frac12 bt^2$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
  time   velocity err   position err
   1 s     0.01 m/s        0.005 m
  10 s     0.10 m/s        0.5 m
  60 s     0.60 m/s       18 m
```

**Nothing changed except the clock.** The $t^2$ is what makes free-running IMU integration useful for seconds, not minutes.

</details>

### Problem 4 — attitude error

Orientation is wrong by 1° in roll. How much of gravity leaks into the horizontal axis?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

$g \sin\theta$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$9.81 \times \sin 1° \approx 0.171$ m/s² — **17× the bias above**, 8.6 m of position error after 10 s. Getting orientation slightly wrong is far worse than a slightly biased accelerometer.

</details>

### Problem 5 — why the IMU is still worth it

A LiDAR scans at 10 Hz; an IMU at 200–1000 Hz. A landing robot can rotate 20° in 100 ms. What does each sensor know about that rotation?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The LiDAR sees one scan before and one after — nothing in between. The IMU samples it hundreds of times. **IMU = fast but drifting; LiDAR = slow but anchored.** LIO combines them.

</details>

### Problem 6 — challenge — two sources compared

A constant 0.02 m/s² bias and a 1° attitude error are both present for 20 s. Which dominates the position error, and by how much?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Bias: $\frac12 \times 0.02 \times 400 = 4$ m. Attitude: $\frac12 \times 0.171 \times 400 \approx 34$ m. **Attitude error dominates by about 8×** — hence the emphasis on gravity alignment and attitude correction.

</details>

---

**You can now:** explain specific force, compute how bias and attitude error drift, and say why IMUs must be combined with an absolute sensor.
