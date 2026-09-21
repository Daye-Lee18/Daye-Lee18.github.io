---
layout: study-chapter
title: "12.2 Fair comparison"
description: "One trajectory, four honest numbers."
importance: 12.2
category: SLAM
series: slam_history
permalink: /study/slam/history/12-2-fair-comparison/
---

> **Goal:** recognise evaluation choices that change a reported error, and list what to check before comparing papers.

---

### Problem 1 — one trajectory, four scores

The same estimated trajectory is scored four ways:

```text
SE(3) alignment, all frames                    0.62 m
Sim(3) alignment (rescaling allowed)           0.15 m
SE(3), the 12 s where tracking was lost dropped 0.11 m
SE(3), first 100 m only                         0.04 m
```

Are any of these dishonest?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

None — every one is correctly computed. The 15× spread comes entirely from **protocol choices**. Two papers both reporting “0.15 m” may not be comparable at all.

</details>

### Problem 2 — name the traps

For each of Sim(3) alignment, dropping failed segments and loose timestamp matching, say how it flatters a result.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

**Sim(3)** hides the scale error monocular systems suffer from. **Dropping failures** rewards a method that gives up over one that keeps trying and drifts. **Loose timestamps** let the aligner pick the most flattering ground-truth pose.

</details>

### Problem 3 — KITTI's choice

KITTI scores sub-trajectories of 100–800 m and reports error as a percentage of distance. What question does that answer?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

“How fast does it drift?” — not “how far off is the whole trajectory?”. Don't treat it as the same metric as a single ATE.

</details>

### Problem 4 — challenge — a checklist

Before comparing two SLAM results, list what must match.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Sensors, dataset segments, compute platform, alignment type (SE(3) vs Sim(3)), scale handling, timestamp association, treatment of failed segments, and metric definition. Report latency, memory and failure rate alongside accuracy.

</details>

---

**You can now:** list the protocol choices that change a reported error and check them before comparing results.
