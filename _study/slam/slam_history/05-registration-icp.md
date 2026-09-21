---
layout: study-chapter
title: "Chapter 5. ICP and point cloud registration"
description: "Understanding iterative optimisation through the problem of aligning two observations."
importance: 5
category: SLAM
series: slam_history
permalink: /study/slam/history/05-registration-icp/
---

This chapter is two short problem sets. Uses the frame transforms from Chapter 4.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                      | You will be able to | Problems                                         |
| :------------------------------------------------------------------------ | :------------------ | :----------------------------------------------- | --- |
| [5.1 The ICP loop]({{ '/study/slam/history/05-1-icp-loop/'                | relative_url }})    | run ICP by hand and see why one outlier ruins it | 6   |
| [5.2 When a low error lies]({{ '/study/slam/history/05-2-low-error-lies/' | relative_url }})    | spot repeated-structure and degeneracy failures  | 4   |

## Original reading

- KRoC registration lecture: the part linking ICP to robust, global and learning-based approaches. Local copy: `_resource/slam/kroc2026/06-registration-hyungtae-lim.pdf`.
- For Grisetti (2016), use the description and links in the recommended guide. The old direct PDF address now returns 404, so it is not included in the local resources.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Registration</dt><dd>Estimating the rigid transform that aligns two observations of the same scene.</dd>
  <dt>ICP</dt><dd>An iterative loop that alternates correspondence search and pose optimisation.</dd>
  <dt>Correspondence</dt><dd>A proposed match between elements in the source and target observations.</dd>
  <dt>Point-to-plane residual</dt><dd>The signed distance from a transformed point to a target surface tangent plane.</dd>
  <dt>Degeneracy</dt><dd>A geometry in which some motion directions are weakly constrained or unobservable.</dd>
</dl>
</aside>
