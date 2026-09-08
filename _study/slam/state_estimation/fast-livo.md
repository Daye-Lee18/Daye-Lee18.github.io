---
layout: study-chapter
title: "FAST-LIVO — paper review"
description: "Combines LiDAR, inertial and image data in a sparse-direct manner; the prior work behind FAST-LIVO2."
category: SLAM
series: state_estimation
importance: 11
permalink: /study/slam/state-estimation/fast-livo/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Combines LiDAR, inertial and image data in a sparse-direct manner; the prior work behind FAST-LIVO2.

| Item           | Detail                                                                                |
| :------------- | :------------------------------------------------------------------------------------ |
| Paper          | FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry      |
| Venue          | IROS 2022                                                                             |
| Source         | [Paper / author material](https://github.com/hku-mars/FAST-LIVO)                      |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below |
| Source checked | 2026-09-07                                                                            |

## 1. The problem it addresses

LiDAR geometry and image information have different strengths, so the aim is to exploit both within the same odometry process.

## 2. Three key points to present

1. **Combining LIO and VIO:** the LiDAR-inertial and visual-inertial subsystems are tightly connected.
2. **Sparse-direct vision:** a sparse-direct approach is used on the image side.
3. **Multi-sensor odometry:** the complementary information provided by images and LiDAR is used together.

Basis for this technical summary: [paper / author description](https://github.com/hku-mars/FAST-LIVO).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
LiDAR + IMU → LIO subsystem ↔ VIO subsystem ← images
```

The official repository describes the architecture as two tightly coupled direct odometry subsystems. Do not retroactively attribute the unified voxel map and sequential update details of FAST-LIVO2 to this paper.

## 4. Experimental results and interpretation

The official repository provides access to the paper, code and data. This page is an introductory review based on the authors' description; per-dataset performance figures are to be recorded after a close reading. [Source](https://github.com/hku-mars/FAST-LIVO)

Getting camera synchronisation and extrinsics right is part of reproduction. Do not record the improvement claims of the later version as experimental results of this earlier paper.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. What state and map information do the LIO and VIO subsystems actually share?
2. How are the points or patches used to compute the image residual selected?
3. Among the improvements in FAST-LIVO2, which are changes in map representation and which are changes in the update scheme?

**Proposed experiment:** first confirm that the same sensor logs can be reproduced. Annotate the image exposure and blur state to separate segments where the image information is usable from those where it is not.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Why must you not confuse describing FAST-LIVO as two tightly coupled subsystems with the unified voxel map of FAST-LIVO2?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The LIO–VIO coupling of the earlier method and the single map representation with sequential update of the later one share data in concretely different ways. Retroactively applying the later paper's improvements to the earlier method leads to a wrong understanding of the states, residuals and map association. The actual interfaces of each paper have to be checked.

</details>

### Q2. Math and reasoning

When combining a LiDAR residual \(r*L\) and an image residual \(r_V\) into one objective \(J=\|r_L\|*{W*L}^2+\|r_V\|*{W_V}^2\), why do the weights matter?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The two residuals have different units, dimensions and noise. Weights that do not match the covariances let one modality numerically dominate. Whitened residuals, robust losses, correlations and the linearisation point all have to be examined. Simply matching the number of residuals is not enough.

</details>

### Q3. Systems and debugging

After adding a camera, the pose drifts slowly even on a static scene. How would you isolate the calibration problem?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Ablate the time offset, extrinsics, camera intrinsics and distortion, and the exposure model one at a time, holding the others fixed. Rotation-centred experiments are useful for time and extrinsic rotation error; targets at several depths for translation and intrinsic error. Look at the LiDAR-only baseline alongside the directionality of the reprojection and photometric residuals.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/hku-mars/FAST-LIVO)

Find the state shared by the LIO and VIO subsystems in the repository and write a data ownership table. If both callbacks can update the state simultaneously, what bugs arise and how would you fix them?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Start from the ROS subscribers and follow symbol references through the LiDAR and IMU queues, the camera queue, the state struct, the map and the publish path. Record the writer, readers, timestamp and lock for each object. Simultaneous updates can cause lost updates, mixing of states from different timestamps, and map–pose inconsistency. Either serialise updates in a single estimator thread, or use a timestamp-ordered event queue with explicit mutex/snapshot ownership. Adding locks alone does not guarantee temporal ordering.

</details>

## 7. Close-reading and presentation record

Use the summary above as a starting point, check the equations, figures and experiment tables in the original, then fill this in yourself. Keep reproduction results you have not yet run separate from the paper's results.

| Item to record           | Personal review notes                                                                |
| :----------------------- | :----------------------------------------------------------------------------------- |
| States, inputs, outputs  | Not written — record frames, units and sensor rates                                  |
| Key equations            | Not written — explain equation numbers, variable meanings, assumptions and residuals |
| Key figures              | Not written — explain the figure number and the data flow in your own words          |
| Experimental evidence    | Not written — table/figure numbers, dataset, baselines, metrics and conditions       |
| Ablation                 | Not written — which element was removed and what changed                             |
| Failure cases and limits | Not written — separate what the authors report from your own inference               |
| Code and reproduction    | Not written — version, configuration, logs, hardware, measurements                   |
| Final judgement          | Not written — reasons to adopt or defer for Vision60                                 |

- [ ] I can explain the three key contributions with evidence from the original paper.
- [ ] I can explain how the states and observations are connected.
- [ ] I have separated the paper's results from the Vision60 application hypotheses.

**Read next:** [FAST-LIVO2 review]({{ '/study/slam/state-estimation/fast-livo2/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
