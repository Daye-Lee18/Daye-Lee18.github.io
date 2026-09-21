---
layout: study-chapter
title: "8.1 Depth from two views"
description: "Why one image cannot give depth, and two can."
importance: 8.1
category: SLAM
series: slam_history
permalink: /study/slam/history/08-1-depth-from-views/
---

> **Goal:** use the pinhole projection, see why a single pixel has no depth, and explain triangulation.

Pinhole camera throughout: $u = f_x X/Z + c_x$, with $f_x = 500$ and $c_x = 320$.

---

### Problem 1 — project a point

Where does the camera-frame point $(2, 0, 10)$ land in $u$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$u = 500 \times 2/10 + 320 = 420$.

</details>

### Problem 2 — depth ambiguity

Points $(0,0,1)$, $(0,0,5)$ and $(0,0,50)$. Which pixels?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

All land on $u = 320$. **One pixel is a direction, not a point** — infinitely many depths along one ray. A single image does not contain the answer.

</details>

### Problem 3 — closer means bigger shift

Now $(2, 0, 5)$. Where does it land, and what does that say about near vs far points?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$u = 520$. Same lateral offset, half the depth, twice the pixel offset. **Near points move more in the image** — the basis of stereo and motion parallax.

</details>

### Problem 4 — two views

Move the camera 1 m sideways and take a second image of the same point. Why does that fix depth?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

From each camera position the point defines a different ray, and **two rays meet at one point**. That is triangulation, and why visual SLAM estimates structure from correspondences across viewpoints.

</details>

### Problem 5 — challenge — pixel error in metres

A 1-pixel error in $u$ corresponds to how many metres of lateral position at $Z = 10$ m? At $Z = 50$ m?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

$\partial u / \partial X = f_x / Z$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$Z/f_x$: 0.02 m at 10 m, 0.1 m at 50 m. **The same pixel noise means 5× more position error far away** — distant features constrain translation weakly.

</details>

---

**You can now:** project points with a pinhole model, explain why one image has no depth, and relate pixel error to metric error.
