---
layout: study-chapter
title: "1.2 Loop closure, and what counts as SLAM"
description: "Turning “I have been here before” into a correction."
importance: 1.2
category: SLAM
series: slam_history
permalink: /study/slam/history/01-2-loop-closure/
---

> **Goal:** explain what a loop closure adds and why it is not the definition of SLAM.

Uses Part 1.1.

---

### Problem 1 — the first absolute fact

In Part 1.1's square, every measurement was relative to the previous moment. The camera now recognises the start. What kind of information is that, and why is it special?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

It relates the pose at $t = 400$ directly to the pose at $t = 0$ — **the first constraint in the whole chain that is not relative to the immediately preceding moment.** That is a **loop closure**.

</details>

### Problem 2 — where does the 0.5 m go?

Option A: move only the final pose back to $(0,0)$. Option B: spread the 0.5 m over the whole trajectory. Which is better, and why?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

B. Option A makes the endpoint right and leaves every intermediate pose — and every landmark drawn from them — wrong. The error happened gradually, so it should be undone gradually. Chapter 6 computes this redistribution exactly.

</details>

### Problem 3 — looks the same ≠ is the same

A first-floor corridor and a second-floor corridor look identical. The robot declares a loop between them. What happens to the map?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The optimiser folds the building in half to make that constraint true. **Recognising a similar-looking place and geometrically verifying it is the same place are different steps**, and skipping the second is how maps get destroyed.

</details>

### Problem 4 — challenge — is it SLAM?

“If you just bolt loop closure onto odometry, can you always call it SLAM?” Answer from the perspective of what is being estimated.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Ask: does the system keep a reusable environment representation, or only correct past poses?

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Not necessarily. SLAM estimates the robot state **and** an environment representation together. A system that only corrects past poses and keeps no reusable map is closer to pose-graph localisation. Conversely, SLAM without loop closure — keeping only a local map — is still SLAM. Decide by the state and map scope, not by whether a loop closure is present.

</details>

---

**You can now:** explain what a loop closure contributes, why error should be spread rather than dumped on the last pose, and why loop closure alone does not define SLAM.
