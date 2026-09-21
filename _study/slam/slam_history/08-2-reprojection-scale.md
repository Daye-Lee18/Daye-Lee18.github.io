---
layout: study-chapter
title: "8.2 Reprojection error and scale"
description: "What bundle adjustment fits, and what a single camera can never know."
importance: 8.2
category: SLAM
series: slam_history
permalink: /study/slam/history/08-2-reprojection-scale/
---

> **Goal:** compute a reprojection error, explain what bundle adjustment adjusts, and show why monocular scale is unobservable.

Same pinhole model as Part 8.1.

---

### Problem 1 — reprojection error

Guessed point $(2, 0, 10)$ projects to $u = 420$, but the feature was observed at $u = 423$. What is the reprojection error? What could be wrong?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

3 pixels. Either the point is slightly misplaced **or** the camera pose is slightly off — one residual cannot tell which.

</details>

### Problem 2 — the bundle

Bundle Adjustment does not fix the point first and then the pose. What does it do?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It nudges **all** points and **all** poses at once to minimise every reprojection error in every frame: $r_{ij} = u_{ij} - \pi(T_{C_iW}P_j)$. A point seen in 20 frames gets 20 such tests.

</details>

### Problem 3 — double everything

Scale the scene: point $(4, 0, 20)$ and double the camera's motion. Where does the point project?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$u = 500 \times 4/20 + 320 = 420$ — identical. **A dollhouse filmed up close and a real house filmed from far away give the same video.**

</details>

### Problem 4 — what fixes scale?

Name three sources of absolute scale for a monocular system.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A stereo baseline (known distance between cameras), an IMU (gravity is 9.81 m/s²), wheel odometry, or a known object size. Learned depth is a prior from training data, not a measurement.

</details>

### Problem 5 — challenge — the evaluation trap

A monocular trajectory is uniformly 1.7× too large. Evaluated with Sim(3) alignment, what error is reported? With SE(3)?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Sim(3) may rescale, so the scale error **vanishes** from the score; SE(3) cannot rescale, so it appears in full. Papers reporting different alignments are not comparable (Chapter 12).

</details>

---

**You can now:** compute reprojection error, explain what bundle adjustment adjusts, and show why monocular scale cannot come from images alone.
