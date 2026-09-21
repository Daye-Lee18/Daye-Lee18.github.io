---
layout: study-chapter
title: "10.2 Deskew, and filter versus smoother"
description: "A scan is not a snapshot; and two ways to fuse."
importance: 10.2
category: SLAM
series: slam_history
permalink: /study/slam/history/10-2-deskew-filter-smoother/
---

> **Goal:** compute scan distortion, explain deskew, and compare filtering with fixed-lag smoothing.

Uses Part 10.1.

---

### Problem 1 — a scan is not a photo

A spinning LiDAR takes 100 ms per scan. The robot moves at 1 m/s. How far apart are the positions where the first and last points were measured?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

0.1 m. Treating them as one instant bends straight walls by 10 cm.

</details>

### Problem 2 — rotation smears more

The robot yaws at 90°/s during one 100 ms scan. How far sideways does a point 10 m away appear to move?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Angle turned × range (in radians).

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

9° ≈ 0.157 rad, times 10 m ≈ 1.57 m. **Rotation distorts far points enormously** — which is why landing impacts wreck point clouds.

</details>

### Problem 3 — deskew

What does deskew do, and what two things must be accurate for it to work?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It moves every point to where it would have been measured at one common reference time, using IMU-derived motion. It needs **accurate per-point timestamps** and an **accurate LiDAR–IMU extrinsic**. Without it, ICP is handed a genuinely bent wall.

</details>

### Problem 4 — two designs

Filter (FAST-LIO2) vs fixed-lag smoother (window of states). Which can revise a state from 2 s ago using a measurement that arrives now?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The smoother: it keeps the window's states and can relinearise them. The filter has marginalised them away. The price is more computation and latency.

</details>

### Problem 5 — challenge — must you choose one?

Does a LIO system have to be a pure filter or a pure smoother?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No. A common design uses a filter for fast propagation and local odometry, and smoothing for a keyframe graph or loop closure. The real axes are latency, state horizon, nonlinearity and the need for global correction — not old versus new.

</details>

---

**You can now:** compute scan distortion, say what deskew needs, and compare filtering with smoothing.
