---
layout: study-chapter
title: "7.2 Fill-in and ordering"
description: "Solving can make a sparse matrix dense — unless you choose the order."
importance: 7.2
category: SLAM
series: slam_history
permalink: /study/slam/history/07-2-fill-in/
---

> **Goal:** count fill-in edges and show that elimination order changes the cost.

Solvers eliminate variables one at a time. Eliminating a variable connects all of its neighbours to each other.

---

### Problem 1 — eliminate a landmark

Landmark $L$ is seen by poses A, B, C (and they are not otherwise linked). Eliminate $L$. Which new edges appear?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

All of $L$'s neighbours become mutually connected.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

A–B, A–C, B–C: three new edges no sensor ever measured. That is **fill-in**.

</details>

### Problem 2 — a popular variable

A variable has 50 neighbours. How many edges can eliminating it add?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

How many pairs among 50 things?

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\binom{50}{2} = 1225$. Eliminating low-degree variables first keeps this small.

</details>

### Problem 3 — order matters

Poses A–B–C are chained by odometry (A–B and B–C exist). Landmark L is seen by A, B, C. Pose D sees only L.

Count fill-in edges for (a) eliminate L first, (b) eliminate D first, then L.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

For each elimination, list which neighbour pairs are not already connected.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

(a) L's neighbours are A, B, C, D. Missing pairs: A–C, A–D, B–D, C–D → **4 new edges**.
(b) D has one neighbour, so 0 new edges. Then L's neighbours are A, B, C; only A–C is missing → **1 new edge**.
Same graph, same answer, 4× less fill-in — purely from order.

</details>

### Problem 4 — challenge — rule of thumb

State a rule of thumb for elimination order, and name a heuristic libraries use.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

**Eliminate cheap, low-degree variables first; postpone the heavily connected ones.** Libraries use ordering heuristics such as COLAMD before touching the numbers.

</details>

---

**You can now:** count fill-in, show that elimination order changes it, and state the low-degree-first rule.
