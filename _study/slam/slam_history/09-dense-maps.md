---
layout: study-chapter
title: "Chapter 9. From sparse maps to dense maps"
description: "What should a map store, and which questions should it answer?"
importance: 9
category: SLAM
series: slam_history
permalink: /study/slam/history/09-dense-maps/
---

This chapter is two short problem sets. Read after Chapter 8.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                                  | You will be able to | Problems                         |
| :------------------------------------------------------------------------------------ | :------------------ | :------------------------------- | --- |
| [9.1 Choosing a map, and what it costs]({{ '/study/slam/history/09-1-choosing-a-map/' | relative_url }})    | pick a map and size a dense grid | 6   |
| [9.2 Occupancy in log-odds]({{ '/study/slam/history/09-2-occupancy-log-odds/'         | relative_url }})    | update an occupancy cell         | 5   |

## Original reading

- KinectFusion original paper: the pipeline figure and the explanation of volumetric integration. Local copy: `_resource/slam/papers/kinectfusion2011.pdf`.
- KRoC 3D World: start from PDF pages 40–43.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Sparse map</dt><dd>A compact set of geometric landmarks mainly useful for localisation and optimisation.</dd>
  <dt>Occupancy map</dt><dd>A probabilistic representation of free, occupied and unknown space.</dd>
  <dt>TSDF</dt><dd>A voxel field storing truncated signed distance to the nearest observed surface.</dd>
  <dt>Sensor model</dt><dd>A probabilistic rule for turning measurements and ray paths into map evidence.</dd>
  <dt>Map representation</dt><dd>A design choice determined by the questions the robot must answer, not by visual density alone.</dd>
</dl>
</aside>
