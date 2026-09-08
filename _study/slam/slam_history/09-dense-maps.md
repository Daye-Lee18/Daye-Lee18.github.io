---
layout: study-chapter
title: "Chapter 9. From sparse maps to dense maps"
description: "What should a map store, and which questions should it answer?"
importance: 9
category: SLAM
series: slam_history
permalink: /study/slam/history/09-dense-maps/
---

> **Goal:** Distinguish sparse point, occupancy and surface representations by purpose.  
> **Workload:** 10–15 minutes. Read after Chapter 8.

## 1. A map that is good for localisation and a map that shows surfaces

Ask two different questions of the same map and you find out they want different maps.

```text
  Q1  "where am I?"                Q2  "can I drive through there?"

  needs: a few points I can        needs: is this specific cubic
         recognise again                  metre empty or not

  200 corner features are          200 corner features tell you
  plenty                           nothing about the gap between them
```

A handful of feature points on the corners of a wall is enough to estimate a camera pose — the localiser only needs things it can re-recognise. But a robot planning a path needs to know about the _space_, and a sparse cloud is silent there:

```text
    ●        ●        ●        ●         four detected corners

  is the region between them a solid wall,
  a doorway, or simply never observed?
  the sparse map cannot say — it has no entry for "between"
```

So the map representation is chosen to match the task it will serve, not to be maximally detailed.

| Representation   | What it stores                             | The first question to ask               |
| ---------------- | ------------------------------------------ | --------------------------------------- |
| Sparse landmarks | Positions of re-observable points          | Can the correspondences be found again? |
| Occupancy grid   | Occupancy likelihood of each cell          | Does it separate free from unobserved?  |
| TSDF             | Truncated signed distance near the surface | How are surfaces fused?                 |
| Mesh / surfels   | Faces or local surface elements            | What does a surface update cost?        |

This comparison is a study checklist for choosing a map. The important thing is not to mistake information a representation does not provide for an estimation result.

## 2. Depth cameras and KinectFusion

[KinectFusion (2011)](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/) is a landmark transition to dense surface reconstruction from a moving depth camera. Rather than keeping each depth frame, it merges every new frame into one shared volume.

The reason to merge is straightforward. A cheap depth sensor is noisy — measure the same flat wall ten times and you get ten different answers:

```text
  frame   measured distance to the wall
    1        2.03 m
    2        1.98 m       each frame: ±3 cm of noise
    3        2.04 m
    ...
   10        1.97 m
            ───────
   average   2.00 m       averaging 10 frames: noise drops by √10
```

Averaging turns a jittery surface into a smooth one, which is why fused reconstructions look so much cleaner than any single depth frame.

The catch is that merging assumes you knew where the camera was. Get the pose wrong by 5 cm and the same wall is written into the volume twice, 5 cm apart:

```text
  pose correct              pose off by 5 cm         object moved
  ────────────              ────────────────         ────────────
  ║                         ║ ║                      ║      ║
  ║  one clean wall         ║ ║  doubled wall        ║      ║  a trail of
  ║                         ║ ║                      ║      ║  where it was
```

Tracking and fusion therefore depend on each other: good poses give a clean model, and a clean model gives good poses.

The [KRoC 3D World lecture](https://drive.google.com/file/d/1OTZjzUGls3fjSQed7LU-xzjS_78e7BIW/view) treats dense/volumetric SLAM alongside feature-based and direct methods.

## 3. Compute the memory yourself

"Just use a finer grid" is the instinct, and this is the calculation that kills it. Assume a simple dense grid over a 10 m cube, 8 bytes per voxel:

```text
  voxel size    voxels per axis    total voxels     memory
  ──────────    ───────────────    ────────────     ──────
    10 cm            100            1,000,000        8 MB
     5 cm            200            8,000,000       64 MB
   2.5 cm            400           64,000,000      512 MB
  1.25 cm            800          512,000,000        4 GB
```

Each row is twice as precise as the one above and **eight times** as expensive, because the cost is cubed: halving the voxel doubles the count along all three axes. Going from 10 cm to 1.25 cm — still not fine enough to see a door handle — takes you from 8 MB to 4 GB.

Scaling the room is just as unforgiving. The same 10 cm voxels over a 100 m building, not a 10 m room, means $1000^3 = 10^9$ voxels, or 8 GB.

This is dense allocation only, and it excludes indices, buffers and sparse data structures. It is also why real systems never allocate the full cube: octrees, hash tables and rolling local volumes exist precisely to avoid paying for the empty space, which is nearly all of it.

## Check questions

### Question 1 — Concept

Explain why an occupancy map has to distinguish `free`, `occupied` and `unknown`. What failure occurs if every space without a point in the point cloud is marked free?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Free is space with observational evidence that a sensor ray passed through it; occupied is space where a return or surface was observed; unknown is space not yet observed or occluded. Turning unknown into free merely because no point landed there can misjudge regions beyond sensor range, behind objects, or on weakly reflective surfaces as safe. Free-space evidence has to be updated explicitly using ray casting and a sensor model that includes minimum/maximum range and invalid returns.

</details>

### Question 2 — Math

An occupancy cell has prior probability $p(m)=0.5$. Two observations, assumed independent, give inverse sensor models $p(m\mid z_1)=0.7$ and $p(m\mid z_2)=0.8$. Find the posterior occupancy probability with a log-odds update.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Log-odds is $l(p)=\log\frac{p}{1-p}$. The log-odds of the prior $0.5$ is 0, so

$$
l=\log\frac{0.7}{0.3}+\log\frac{0.8}{0.2}
=\log\left(\frac{28}{3}\right)\approx2.234.
$$

Converting back to a probability,

$$
p=\frac{1}{1+e^{-l}}=\frac{28}{31}\approx0.903.
$$

In practice consecutive scans are not fully independent because of pose error and repeated observation. Log-odds clamping and sensor-model validation are needed so the same information does not make the map overconfident.

</details>

## Original reading

- KinectFusion original paper: the pipeline figure and the explanation of volumetric integration. Local copy: `_resource/slam/papers/kinectfusion2011.pdf`.
- KRoC 3D World: start from PDF pages 40–43.
