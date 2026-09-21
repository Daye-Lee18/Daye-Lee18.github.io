---
layout: study-chapter
title: "Chapter 10. From IMU fusion to LIO"
description: "Why fast inertial prediction and external observations need each other."
importance: 10
category: SLAM
series: slam_history
permalink: /study/slam/history/10-inertial-lio/
---

This chapter is two short problem sets and the doorway to the LIO part.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                                            | You will be able to | Problems                                 |
| :---------------------------------------------------------------------------------------------- | :------------------ | :--------------------------------------- | --- |
| [10.1 What an accelerometer measures]({{ '/study/slam/history/10-1-accelerometer/'              | relative_url }})    | compute IMU drift from bias and attitude | 6   |
| [10.2 Deskew, and filter versus smoother]({{ '/study/slam/history/10-2-deskew-filter-smoother/' | relative_url }})    | size scan distortion and compare designs | 5   |

## Original reading

- KRoC IMU: PDF pages 13–24. Local copy: `_resource/slam/kroc2026/05-imu-giseop-kim.pdf`.
- Read only the abstract and system overview of the two LIO papers. Detailed derivations and code analysis are picked up in the LIO part.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>IMU propagation</dt><dd>High-rate prediction of orientation, velocity and position by integrating inertial measurements.</dd>
  <dt>Bias</dt><dd>A systematic sensor offset whose integration causes rapidly growing state error.</dd>
  <dt>Deskew</dt><dd>Compensating each LiDAR point for motion during acquisition of a scan.</dd>
  <dt>LIO</dt><dd>Tightly coupled LiDAR–inertial odometry that combines fast inertial prediction with geometric correction.</dd>
  <dt>Filtering vs smoothing</dt><dd>Recursive current-state updates versus joint optimisation over a retained state history.</dd>
</dl>
</aside>
