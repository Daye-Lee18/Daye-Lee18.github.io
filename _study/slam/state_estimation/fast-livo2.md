---
layout: study-chapter
title: "FAST-LIVO2 — paper review"
description: "A LIVO that links LiDAR geometry and image intensity in the same voxel map and fuses them sequentially in an ESIKF."
category: SLAM
series: state_estimation
importance: 5
permalink: /study/slam/state-estimation/fast-livo2/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** A LIVO that links LiDAR geometry and image intensity in the same voxel map and fuses them sequentially in an ESIKF.

| Item           | Detail                                                                                |
| :------------- | :------------------------------------------------------------------------------------ |
| Paper          | FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry                               |
| Venue          | T-RO 2025 · online publication 2024                                                   |
| Source         | [Paper / author material](https://arxiv.org/abs/2408.14035)                           |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below |
| Source checked | 2026-09-07                                                                            |

## 1. The problem it addresses

LiDAR and images differ in observation representation and dimension. The two have to be combined efficiently so that each supplies the constraints the other lacks.

## 2. Three key points to present

1. **Sequential update:** LiDAR and image observations are incorporated sequentially in an ESIKF.
2. **Unified voxel map:** image patches are attached to LiDAR points so that geometric and image registration share the same reference.
3. **Stronger image alignment:** plane priors, reference patch updating, raycasting and exposure estimation are used.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2408.14035).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
IMU prediction → LiDAR geometric update → image photometric update → unified voxel map and pose
```

LiDAR uses raw-point registration and images use photometric error. The flow above is a conceptual summary; check the actual asynchronous sensor schedule in the implementation. Keep the quality of the output map distinct from whether global loop correction is present.

## 4. Experimental results and interpretation

Beyond benchmark and in-house data comparisons and module validation, the authors present UAV onboard navigation, aerial mapping and 3D rendering applications. These are not a direct validation of Vision60's landing and slip conditions. [Source](https://arxiv.org/abs/2408.14035)

The contribution of the images depends on the observable texture, exposure and blur. Adding a camera also brings the cost of time synchronisation and inter-sensor calibration.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. In segments where LiDAR degenerates, is there usable visual information left for the camera?
2. How much motion blur and exposure variation occur during gait vibration and at the moment of landing?
3. In a LiDAR-only versus LIVO comparison, have the sensor timing conditions been matched?

**Proposed experiment:** split the segments with weak geometry into bright, dark and blurred segments and compare error, failure rate and computation between LiDAR-only and image fusion.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Explain why FAST-LIVO2 links LiDAR and images in a single voxel map, and the benefit of the sequential update.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

LiDAR points build the geometric structure, and attaching image patches to those points gives both modalities the same 3D reference. The sequential update lets the LiDAR geometric residual and the image photometric residual, which have different dimensions and models, be handled separately while still updating the same state. That said, the independence and linearisation assumptions and the effect of update order have to be checked.

</details>

### Q2. Math and reasoning

Decompose the pose Jacobian of the photometric residual \(r(u)=I*k(\pi(TP))-I*{ref}(u)\) with the chain rule. On which images does the information become small?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

\(\partial r/\partial\xi=\nabla I_k\,(\partial\pi/\partial P')\,(\partial P'/\partial\xi)\): the product of the image gradient, the projection Jacobian and the SE(3) motion Jacobian. On textureless surfaces \(\nabla I\) is small; saturation and exposure changes break brightness constancy; and depending on depth and geometric layout, sensitivity to certain motions weakens.

</details>

### Q3. Systems and debugging

LIVO improves ATE over LiDAR-only, but the failure rate on landing has gone up. Suggest possible causes and ablations.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Suspect motion blur, rolling shutter, exposure changes, camera–IMU time error, extrinsic flex, and the compute latency of the image update. On the same log, compare LiDAR update only, visual update only, excluding blurred frames, exposure estimation on/off, and a timestamp offset sweep. Look at mean ATE separately from the failure rate and worst-case latency on impact segments.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/hku-mars/FAST-LIVO2)

In `src/LIVMapper.cpp`, trace what the measurement buffer decides when the image time is ahead of or behind the latest LiDAR and IMU times. How would you build a test that distinguishes an image drop from a wait?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Tabulate `img_time_buffer`, the LiDAR scan start and end, the time of the last LIO update, the latest IMU time, and `exposure_time_init`. An image older than an already-processed LIO time should be dropped; if the required LiDAR/IMU data has not yet arrived, the buffer should be kept and the system should wait. Inject synthetic message times in various orders and verify the buffer size, the return status, `lio_vio_flg` and the processing timestamps. Also check that there is no early return while a mutex is held.

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

**Read next:** [FAST-LIVO review]({{ '/study/slam/state-estimation/fast-livo/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
