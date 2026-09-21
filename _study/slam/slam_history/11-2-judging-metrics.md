---
layout: study-chapter
title: "11.2 Judging the numbers"
description: "Pretty pictures and depth scores are not pose accuracy."
importance: 11.2
category: SLAM
series: slam_history
permalink: /study/slam/history/11-2-judging-metrics/
---

> **Goal:** compute a depth metric and explain why it does not measure trajectory accuracy.

---

### Problem 1 — AbsRel

A depth net outputs $1.2\times$ every true depth. Compute $\mathrm{AbsRel} = \frac1N\sum |\hat d - d|/d$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Every term is $0.2$, so AbsRel = 0.2 (20%).

</details>

### Problem 2 — same score, different error

True depths $[2, 5, 10]$ m. Another net adds a constant +0.5 m. AbsRel?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$(0.25 + 0.1 + 0.05)/3 \approx 0.133$. A constant offset punishes near points; a scale error punishes all equally. **One number hides the error's shape.**

</details>

### Problem 3 — what AbsRel misses

Why can a good AbsRel still give poor monocular SLAM trajectories?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It says nothing about frame-to-frame consistency, correspondence, camera pose, loop closure or drift. A constant scale bias may even vanish under Sim(3) trajectory alignment.

</details>

### Problem 4 — rendering vs robot

A rendering paper reports PSNR 32 dB. What does a robot need to know that this number does not tell you?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Whether the doorway is at 4.2 m or 4.5 m, and whether a dark region is wall or free space. **A beautiful image can be 20 cm out of place.**

</details>

---

**You can now:** compute AbsRel and explain why depth and rendering metrics do not measure what a robot needs.
