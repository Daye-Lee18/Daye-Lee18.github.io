---
layout: study-chapter
title: "Chapter 1. Why do we need SLAM?"
description: "The relationship between pose and map, odometry versus SLAM, and the broad arc of the field."
importance: 1
category: SLAM
series: slam_history
permalink: /study/slam/history/01-why-slam/
---

This chapter is two short problem sets. No prior knowledge needed.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                                       | You will be able to | Problems                          |
| :----------------------------------------------------------------------------------------- | :------------------ | :-------------------------------- | --- |
| [1.1 Relative measurements and drift]({{ '/study/slam/history/01-1-relative-measurements/' | relative_url }})    | say why odometry error only grows | 6   |
| [1.2 Loop closure, and what counts as SLAM]({{ '/study/slam/history/01-2-loop-closure/'    | relative_url }})    | explain what a loop closure fixes | 4   |

## Original reading

- KRoC History slides: the `What is SLAM?` and `History of SLAM` sections. Local copy: `_resource/slam/kroc2026/01-history-ayoung-kim.pdf`.
- Cadena et al. (2016): read only the Introduction first. Local copy: `_resource/slam/papers/cadena2016-slam-survey.pdf`.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Loop closure</dt><dd>Recognising a previously visited place and adding a constraint that can correct accumulated drift.</dd>
  <dt>Local map</dt><dd>A bounded, nearby representation used for fast and stable short-range estimation.</dd>
  <dt>SLAM</dt><dd>Estimating the robot state and an environment representation together because each depends on the other.</dd>
  <dt>Global map</dt><dd>A persistent representation that connects observations across the full trajectory or operating area.</dd>
  <dt>Odometry</dt><dd>Estimating relative motion over time; errors accumulate unless another source provides correction.</dd>
</dl>
</aside>
