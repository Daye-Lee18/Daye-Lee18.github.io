---
layout: study-chapter
title: "9.2 Occupancy in log-odds"
description: "Free, occupied, unknown — and how evidence adds up."
importance: 9.2
category: SLAM
series: slam_history
permalink: /study/slam/history/09-2-occupancy-log-odds/
---

> **Goal:** update an occupancy cell with log-odds and explain why unknown must not be treated as free.

---

### Problem 1 — three states

A LiDAR sees no points in some region. A teammate marks it free. Why can that be dangerous?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No points can mean beyond range, behind an obstacle, or a dark surface that returned nothing. **Free** needs evidence that a ray passed through; **unknown** means no evidence. Marking unknown as free can send the robot into an obstacle.

</details>

### Problem 2 — log-odds

Log-odds is $l(p) = \log\frac{p}{1-p}$. What is $l$ for $p = 0.5$? For $p = 0.7$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$l(0.5) = 0$ — no information. $l(0.7) = \log(0.7/0.3) \approx 0.847$. Positive means “probably occupied”.

</details>

### Problem 3 — evidence adds

Prior 0.5. Two independent readings say 0.7 and 0.8 occupied. Add their log-odds and convert back to a probability.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

$p = 1/(1 + e^{-l})$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$l = \log(0.7/0.3) + \log(0.8/0.2) = \log(28/3) \approx 2.234$, so $p = 28/31 \approx 0.903$.
**In log-odds, independent evidence just adds** — that is why maps store $l$, not $p$.

</details>

### Problem 4 — conflicting evidence

A third reading then says the cell is free with $p = 0.3$. New probability?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$l \approx 2.234 + \log(0.3/0.7) \approx 1.386$, so $p = 0.8$. The free reading pulls it back but does not erase the earlier evidence.

</details>

### Problem 5 — challenge — overconfidence

Consecutive scans of a static wall are not truly independent. What goes wrong if you add all of them, and what do systems do about it?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The same information is counted many times, so cells become **overconfident** and can no longer change when the world does. Systems **clamp** log-odds to a range and use a validated sensor model.

</details>

---

**You can now:** explain free vs unknown, update occupancy in log-odds, and say why log-odds must be clamped.
