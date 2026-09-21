---
layout: study-chapter
title: "11.1 Where learning enters"
description: "“Deep SLAM” is not one thing."
importance: 11.1
category: SLAM
series: slam_history
permalink: /study/slam/history/11-1-where-learning-enters/
---

> **Goal:** locate which pipeline stage a learning-based method replaces, and what geometry it keeps.

---

### Problem 1 — the pipeline

List the stages of a visual SLAM pipeline from image to map.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

① detect features → ② describe them → ③ match → ④ estimate depth → ⑤ optimise poses → ⑥ map representation. A network could sit at any of these.

</details>

### Problem 2 — classify

Which stage does each replace? SuperPoint; a monocular depth network; DROID-SLAM's recurrent update; NeRF / 3D Gaussian Splatting.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

SuperPoint → ①② (features). Depth net → ④. DROID-SLAM → ⑤ (a network proposes updates, dense BA still enforces geometry). NeRF / 3DGS → ⑥ (the map becomes network weights).

</details>

### Problem 3 — what geometry remains?

A method learns only ①②. Another learns ⑤ and ⑥. Which keeps more of classical bundle adjustment?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The first keeps all of BA; the second has replaced most of it. **Always ask which variables are still optimised geometrically** — that row of the analysis table tells you the risk profile.

</details>

### Problem 4 — reconstruction ≠ SLAM

A NeRF paper shows photorealistic scenes, with camera poses solved offline by COLMAP beforehand. Has it shown it can do SLAM?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No. **Poses were an input, not an unknown**, the whole sequence was available, and time was offline. SLAM must estimate poses online from data seen so far.

</details>

### Problem 5 — challenge — a new city

A learned visual SLAM method beats classical ones on its training data but fails badly in a new city. How would you investigate?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Check distribution differences (appearance, intrinsics, motion, weather, dynamic objects). Ablate by swapping learned modules for classical ones to find which breaks. Measure geometric consistency and failure rate, not just mean ATE. Evaluate leave-one-city-out, controlling lighting, blur and intrinsics.

</details>

---

**You can now:** place a learning-based method in the pipeline and say what geometry it keeps.
