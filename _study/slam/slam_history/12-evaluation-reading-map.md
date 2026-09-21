---
layout: study-chapter
title: "Chapter 12. Evaluation and where to study next"
description: "How to connect the history to performance tables and code reading."
importance: 12
category: SLAM
series: slam_history
permalink: /study/slam/history/12-evaluation-reading-map/
---

This chapter is two short problem sets. The last chapter of History.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                  | You will be able to | Problems                           |
| :-------------------------------------------------------------------- | :------------------ | :--------------------------------- | --- |
| [12.1 ATE and RPE]({{ '/study/slam/history/12-1-ate-rpe/'             | relative_url }})    | compute trajectory metrics by hand | 6   |
| [12.2 Fair comparison]({{ '/study/slam/history/12-2-fair-comparison/' | relative_url }})    | spot evaluation traps              | 4   |

## Where to go next

Continue in [LIO & State Estimation]({{ '/study/slam/state-estimation/' | relative_url }}) in the order IMU model → deskew → error-state → LIO-SAM/FAST-LIO2. That page is the comparison table and index that lays out the scope of the study.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>ATE</dt><dd>Global trajectory disagreement after a stated alignment procedure.</dd>
  <dt>RPE</dt><dd>Relative motion error over a stated time or distance interval, useful for measuring local drift.</dd>
  <dt>SE(3) vs Sim(3)</dt><dd>Rigid alignment preserves scale, while similarity alignment can also remove a global scale error.</dd>
  <dt>Fair comparison</dt><dd>Matching sensors, data segments, compute, alignment and metric definitions before ranking systems.</dd>
  <dt>Operational robustness</dt><dd>Evaluating failures, recovery, latency and resource growth alongside average accuracy.</dd>
</dl>
</aside>
