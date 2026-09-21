---
layout: study-chapter
title: "12.1 ATE and RPE"
description: "Two trajectory metrics, computed by hand."
importance: 12.1
category: SLAM
series: slam_history
permalink: /study/slam/history/12-1-ate-rpe/
---

> **Goal:** compute ATE and RPE for a small trajectory and say what each reveals.

---

### Problem 1 — the data

Estimate $(0,0), (1.1,0), (2.2,0)$; ground truth $(0,0), (1,0), (2,0)$. Position error at each pose?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$0,\ 0.1,\ 0.2$ m.

</details>

### Problem 2 — ATE

Absolute Trajectory Error is the RMSE of those errors. Compute it.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\sqrt{(0 + 0.01 + 0.04)/3} \approx 0.129$ m.

</details>

### Problem 3 — RPE

Relative Pose Error compares each step's motion: estimated 1.1 m per 1 m step. RMSE over the two steps?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Both steps are 0.1 m off, so RPE = 0.1 m.

</details>

### Problem 4 — what each says

Which metric shows the drift per step, and which shows how it piled up?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

**RPE** shows the 10 cm of drift per step. **ATE** shows its accumulation into 20 cm at the end. RPE = local drift rate; ATE = global agreement after alignment.

</details>

### Problem 5 — a spike

A trajectory is perfect except the middle pose is 0.1 m off: errors $0, 0.1, 0$. ATE? Is it the same kind of error as Problem 2?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\sqrt{0.01/3} \approx 0.058$ m. Lower ATE, but it is a jump rather than steady drift — **ATE alone cannot tell them apart**, which is why you report both.

</details>

### Problem 6 — challenge — which to deploy?

Paper A has smaller ATE; paper B has smaller RPE and failure rate. Which would you put on a long-duration robot?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Don't pick on one number. Re-evaluate under the same sensors, compute, data and alignment. For long-duration robots, failure rate, relocalisation time, false loops, memory growth and latency tails often matter more than mean ATE — B may well be the better product choice.

</details>

---

**You can now:** compute ATE and RPE by hand and say what each reveals about drift.
