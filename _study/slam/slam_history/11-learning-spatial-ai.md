---
layout: study-chapter
title: "Chapter 11. Learning-based SLAM and Spatial AI"
description: "Which part of the pipeline does learning actually change?"
importance: 11
category: SLAM
series: slam_history
permalink: /study/slam/history/11-learning-spatial-ai/
---

> **Goal:** Distinguish learned features, learned depth and neural maps.  
> **Workload:** 10–15 minutes. This chapter is about components, not the latest ranking.

## 1. “Deep learning SLAM” is not one thing

“It uses deep learning” tells you almost nothing, because a SLAM pipeline has five or six places a network could sit — and swapping one of them is a completely different system from swapping another.

Lay the pipeline out and mark where learning can enter:

```text
  image → ① find features → ② describe them → ③ match → ④ estimate depth
                                                          ↓
        map representation ⑥ ← ⑤ optimise poses ←─────────┘

  ① ② learned features      SuperPoint, and similar
       (the geometry after them is completely unchanged)

  ④   learned depth         MiDaS, and similar
       (supplies the scale a monocular camera cannot, see Ch. 8 §4)

  ⑤   learned pose update   DROID-SLAM
       (a network proposes the update; BA still enforces geometry)

  ⑥   learned map           NeRF, 3D Gaussian Splatting
       (the map is now network weights, not points)
```

Two papers can both be called "deep SLAM" while sharing no component at all. So before memorising model names, write down for each one: what goes in, what comes out, and **which variables are still being optimised geometrically**. A system that learns ① and ② keeps all of classical BA; a system that learns ⑤ and ⑥ has replaced most of it.

The [KRoC AI Visual SLAM lecture](https://drive.google.com/file/d/1-FZ207zXWqZiEudDnd5EEAaybWdIJzNe/view) separates learned features, depth and neural SLAM along these lines.

[DROID-SLAM](https://arxiv.org/abs/2108.10869) combines recurrent updates with Dense Bundle Adjustment to update poses and pixelwise depth. It is a case of using learning while retaining a geometric optimisation structure. Do not generalise the performance of this one case to all learning-based approaches or all environments.

## 2. A good-looking map and localisation performance

NeRF represents a scene as a radiance field, and 3D Gaussian Splatting renders using Gaussian primitives. The renderings are photorealistic, and it is tempting to read that as "the map is excellent". Be careful: **the metric the pictures are scored on is not the metric a robot cares about.**

```text
  what a rendering paper reports     what a robot needs to know

  PSNR 32 dB                         is the doorway at x = 4.2 or x = 4.5?
  "looks indistinguishable"          is that dark region a wall or free space?

  a photo can be beautiful and still be 20 cm out of place
```

The second confusion is about the inputs. Most striking NeRF and 3DGS results are _reconstructions_: the camera poses were already solved beforehand, usually by COLMAP running offline over the whole sequence.

```text
  reconstruction                     SLAM
  ──────────────                     ──────────────────────
  poses: given as input              poses: an unknown to solve
  data:  the whole sequence          data:  only what has arrived so far
  time:  offline, hours allowed      time:  online, now
```

Those are different problems. A method that produces a gorgeous scene given correct poses has not shown it can estimate poses at all. For the basics of these representations, read the [KRoC 3D Vision lecture](https://drive.google.com/file/d/1mL52klpHEYU6e-yZk3guMaJocLthSAA7/view).

## 3. A small paper-analysis exercise

Fill in one row of the following table for each new paper.

| Item               | What to record                                                 |
| ------------------ | -------------------------------------------------------------- |
| Input              | RGB only, or depth/IMU/pose too?                               |
| What is learned    | Correspondence, depth, update or map?                          |
| Remaining geometry | Where are projection, pose optimisation and loop verification? |
| Runtime conditions | GPU, resolution, memory, pretraining data?                     |
| Output             | Trajectory, surface or semantic labels?                        |

Filled in for DROID-SLAM, the table looks like this:

```text
  Input               RGB only (monocular, stereo or RGB-D variants exist)
  What is learned     the update step — a recurrent net proposes
                      corrections to depth and pose
  Remaining geometry  a lot. Dense Bundle Adjustment still enforces
                      the projection equations from Chapter 8
  Runtime conditions  GPU, and a substantial one; memory grows with
                      the number of keyframes
  Output              trajectory + per-pixel depth
```

The "remaining geometry" row is the informative one. DROID-SLAM did not replace BA with a network; it used a network to _drive_ BA. That is a very different risk profile from a system that regresses poses directly, because the geometric layer still rejects physically impossible answers.

Fill in the same five rows for each new paper and the marketing evaporates fairly quickly.

## Check questions

### Question 1 — Concept

A learning-based Visual SLAM method beats classical approaches on its training data but fails badly in a new city. How would you analyse the cause and propose further experiments?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

First check the differences in appearance, camera intrinsics, motion, weather and dynamic-object distribution between the training data and the new city. Run an ablation that swaps in classical components to identify which module breaks down: learned features, depth, pose update or map representation. Measure geometry consistency, uncertainty calibration and catastrophic failure rate alongside mean ATE. Evaluate leave-one-domain-out across several cities, and isolate causes with experiments controlling illumination, blur and intrinsics. If test-time adaptation is used, ground-truth leakage, computation and online stability also have to be reported.

</details>

### Question 2 — Math and evaluation

Suppose a depth estimator outputs $\hat d_i=1.2d_i$ for every true depth $d_i$. Compute the absolute relative error

$$
\mathrm{AbsRel}=\frac1N\sum_i\frac{|\hat d_i-d_i|}{d_i}
$$

and explain why this result does not adequately account for the pose accuracy of monocular SLAM.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

At each point

$$
\frac{|1.2d_i-d_i|}{d_i}=0.2,
$$

so $\mathrm{AbsRel}=0.2$, that is 20%. But this value summarises only the global scale bias of the depth; it does not directly measure temporal consistency between frames, correspondence, camera pose, loop closure or drift. Aligning a monocular trajectory with Sim(3) may even remove a constant scale bias from the evaluation. Depth metrics, trajectory metrics and failure rate have to be read together.

</details>

## Original reading

- KRoC AI Visual SLAM: PDF pages 31–42. Local copy: `_resource/slam/kroc2026/07-ai-visual-slam-alex-lee.pdf`.
- DROID-SLAM: the architecture figure. Local copy: `_resource/slam/papers/droid-slam2021.pdf`.
- KRoC 3D Vision: PDF pages 59–63 and 81–86. You do not need to read every neural rendering equation on a first pass.
