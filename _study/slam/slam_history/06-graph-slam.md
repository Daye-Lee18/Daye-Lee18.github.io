---
layout: study-chapter
title: "Chapter 6. Graph SLAM and loop closure"
description: "A constraint-based representation in which past trajectory can still be corrected."
importance: 6
category: SLAM
series: slam_history
permalink: /study/slam/history/06-graph-slam/
---

This chapter is two short problem sets. Uses the residual idea from Chapter 5.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                           | You will be able to | Problems                        |
| :----------------------------------------------------------------------------- | :------------------ | :------------------------------ | --- |
| [6.1 Constraints and least squares]({{ '/study/slam/history/06-1-constraints/' | relative_url }})    | solve a pose graph by hand      | 6   |
| [6.2 Wrong loops]({{ '/study/slam/history/06-2-wrong-loops/'                   | relative_url }})    | explain why low cost ≠ true map | 3   |

## Original reading

- KRoC Back-end: the part going from probability to least squares. Local copy: `_resource/slam/kroc2026/04-backend-younggun-cho.pdf`.
- For Stachniss (2016), use the description and links in the recommended guide. The old direct PDF address now returns 404, so it is not included in the local resources.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Factor graph</dt><dd>A graph whose variables are connected by measurement or prior constraints.</dd>
  <dt>Front-end</dt><dd>The component that turns sensor data into proposed correspondences and constraints.</dd>
  <dt>Back-end</dt><dd>The component that optimises the variables to best satisfy all weighted constraints.</dd>
  <dt>Loop closure</dt><dd>A revisit constraint that lets the back-end redistribute accumulated error across past poses.</dd>
  <dt>Robust kernel</dt><dd>A loss function that reduces, but cannot guarantee removal of, the influence of outliers.</dd>
</dl>
</aside>
