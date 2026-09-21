---
layout: study-chapter
title: "Chapter 11. Learning-based SLAM and Spatial AI"
description: "Which part of the pipeline does learning actually change?"
importance: 11
category: SLAM
series: slam_history
permalink: /study/slam/history/11-learning-spatial-ai/
---

This chapter is two short problem sets about components, not rankings.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                              | You will be able to | Problems                                |
| :-------------------------------------------------------------------------------- | :------------------ | :-------------------------------------- | --- |
| [11.1 Where learning enters]({{ '/study/slam/history/11-1-where-learning-enters/' | relative_url }})    | locate what a learned method replaces   | 5   |
| [11.2 Judging the numbers]({{ '/study/slam/history/11-2-judging-metrics/'         | relative_url }})    | see through depth and rendering metrics | 4   |

## Original reading

- KRoC AI Visual SLAM: PDF pages 31–42. Local copy: `_resource/slam/kroc2026/07-ai-visual-slam-alex-lee.pdf`.
- DROID-SLAM: the architecture figure. Local copy: `_resource/slam/papers/droid-slam2021.pdf`.
- KRoC 3D Vision: PDF pages 59–63 and 81–86. You do not need to read every neural rendering equation on a first pass.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Learned component</dt><dd>A network that replaces or supports a specific pipeline stage such as features, depth or pose estimation.</dd>
  <dt>Learned depth</dt><dd>A data-driven depth prediction that still requires checks for scale and geometric consistency.</dd>
  <dt>Neural map</dt><dd>An implicit learned scene representation queried for geometry, appearance or rendering.</dd>
  <dt>Domain shift</dt><dd>Performance loss when deployment conditions differ from the training distribution.</dd>
  <dt>Spatial AI</dt><dd>Extending geometric estimation toward persistent, semantic and task-aware scene understanding.</dd>
</dl>
</aside>
