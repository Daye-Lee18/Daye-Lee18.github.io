---
layout: study-chapter
title: "9.1 Choosing a map, and what it costs"
description: "Different questions need different maps."
importance: 9.1
category: SLAM
series: slam_history
permalink: /study/slam/history/09-1-choosing-a-map/
---

> **Goal:** match a map representation to the question the robot asks, and compute what a dense grid costs.

---

### Problem 1 — two questions

Question A: “where am I?” Question B: “can I drive through there?” Does a map of 200 detected corner features answer both?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It answers A — the localiser only needs things it can recognise again. It cannot answer B: **a sparse map has no entry for the space between features**, so a doorway, a wall and unobserved space all look the same.

</details>

### Problem 2 — match the representation

Match each question to a map: sparse landmarks, occupancy grid, TSDF.
(1) Can I find those correspondences again? (2) Is this cell free, occupied or unknown? (3) Where exactly is the surface?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

(1) sparse landmarks, (2) occupancy grid, (3) TSDF. **Choose the map for the question, not for how dense it looks.**

</details>

### Problem 3 — dense grid memory

A 10 m cube at 10 cm voxels, 8 bytes per voxel. How many voxels, how much memory?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$100^3 = 10^6$ voxels, 8 MB.

</details>

### Problem 4 — finer

Same cube at 5 cm, then 2.5 cm. Memory each?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Halving the voxel doubles the count along **each** of three axes.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

5 cm: $200^3 \times 8$ B = 64 MB. 2.5 cm: 512 MB. **Twice the precision costs eight times the memory.** That is why real systems use octrees, hash tables or rolling local volumes.

</details>

### Problem 5 — why fuse depth frames?

A depth camera measures a wall with ±3 cm noise per frame. Averaging 10 frames reduces the noise to about what?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Averaging $N$ independent measurements divides the standard deviation by $\sqrt{N}$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$3/\sqrt{10} \approx 0.95$ cm. That is why fused reconstructions (KinectFusion) look much cleaner than any single frame.

</details>

### Problem 6 — challenge — fusion needs poses

The fused volume assumes each frame's camera pose is right. What happens if one pose is 5 cm off? If an object moves?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A 5 cm pose error writes the same wall twice, 5 cm apart — a **doubled surface**. A moving object leaves a **trail**. Tracking and fusion depend on each other: good poses give a clean model, a clean model gives good poses.

</details>

---

**You can now:** choose a map by the question it must answer, compute dense-grid memory, and explain why fusion needs accurate poses.
