---
layout: study-chapter
title: "6.2 Wrong loops"
description: "A low cost does not mean a true map."
importance: 6.2
category: SLAM
series: slam_history
permalink: /study/slam/history/06-2-wrong-loops/
---

> **Goal:** explain why optimisation cannot detect false constraints and what the front-end and back-end each do about them.

Uses Part 6.1.

---

### Problem 1 — the solver obeys

A false loop links a first-floor corridor to the second floor. After optimisation the cost went **down**. Is the map better?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

No — the building was folded in half to satisfy the false constraint. **A low cost means the constraints agree with each other, not that they are true.**

</details>

### Problem 2 — whose job?

Split responsibilities: which part of the system asks “why was this constraint created?” and which asks “how were the given constraints reconciled?”

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The first is the **front-end** (proposing and verifying constraints). The second is the **back-end** (optimisation). Most catastrophic SLAM failures are front-end mistakes the back-end faithfully executed.

</details>

### Problem 3 — challenge — is a robust kernel enough?

A single false loop enters. Does applying a robust kernel fully solve it? Answer from both front-end and back-end perspectives.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Not guaranteed. A robust kernel down-weights large residuals, but near the initial guess a false loop may have a _small_ residual or an over-large $\Omega$ and still dominate.
Front-end: geometric verification, temporal/spatial consistency checks. Back-end: robust loss, switchable constraints, DCS, consistency tests. **You need both** — reject first, then limit the influence of what gets through.

</details>

---

**You can now:** explain why optimisation cannot catch a false loop and what the front-end and back-end each contribute.
