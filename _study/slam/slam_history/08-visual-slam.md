---
layout: study-chapter
title: "Chapter 8. Visual SLAM and Bundle Adjustment"
description: "Refining poses and 3D points together from camera observations."
importance: 8
category: SLAM
series: slam_history
permalink: /study/slam/history/08-visual-slam/
---

This chapter is two short problem sets. Read Chapters 4 and 6 first.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                                 | You will be able to | Problems                                            |
| :----------------------------------------------------------------------------------- | :------------------ | :-------------------------------------------------- | --- |
| [8.1 Depth from two views]({{ '/study/slam/history/08-1-depth-from-views/'           | relative_url }})    | explain why one image has no depth                  | 5   |
| [8.2 Reprojection error and scale]({{ '/study/slam/history/08-2-reprojection-scale/' | relative_url }})    | compute reprojection error and spot scale ambiguity | 5   |

## Original reading

- ORB-SLAM: the system overview figure and the description of the three threads. Local copy: `_resource/slam/papers/orb-slam2015.pdf`.
- KRoC 3D Vision: PDF pages 31–38. For the BA text, start with the Introduction and the cost function.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Reprojection error</dt><dd>The pixel difference between an observed feature and the projection predicted by the current pose and map.</dd>
  <dt>Bundle adjustment</dt><dd>Joint nonlinear optimisation of camera poses and 3D landmarks.</dd>
  <dt>Keyframe</dt><dd>A selected camera frame retained for mapping, optimisation and place recognition.</dd>
  <dt>Monocular scale</dt><dd>A gauge freedom: image geometry alone cannot determine absolute metric scale.</dd>
  <dt>Tracking and mapping</dt><dd>Estimating the current camera motion while maintaining and refining the scene representation.</dd>
</dl>
</aside>
