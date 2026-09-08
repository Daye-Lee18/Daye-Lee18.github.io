---
layout: study-chapter
title: "FAST-LIO2 — paper review"
description: "Registers raw LiDAR points directly against a local map and fuses them with IMU in an iterated filter; the baseline of the current Vision60 system."
category: SLAM
series: state_estimation
importance: 2
permalink: /study/slam/state-estimation/fast-lio2/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Registers raw LiDAR points directly against a local map and fuses them with IMU in an iterated filter; the baseline of the current Vision60 system.

| Item           | Detail                                                                                |
| :------------- | :------------------------------------------------------------------------------------ |
| Paper          | FAST-LIO2: Fast Direct LiDAR-inertial Odometry                                        |
| Venue          | T-RO 2022 · arXiv 2021                                                                |
| Source         | [Paper / author material](https://arxiv.org/abs/2107.06829)                           |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below |
| Source checked | 2026-09-07                                                                            |

## 1. The problem it addresses

Designing feature extraction for each scan pattern and managing a growing point cloud map increases the computational cost. FAST-LIO2 changes the registration method and the map data structure together.

## 2. Three key points to present

1. **Direct registration:** raw points are registered to the map without pre-selecting edge and planar feature points. Correspondences and geometric constraints are still required.
2. **Iterated filter fusion:** inertial and LiDAR information are combined on the efficient tightly coupled iterated Kalman filter of the FAST-LIO family.
3. **ikd-Tree:** supports point insertion, deletion, rebalancing and downsampling, updating the local map while moving. The new contributions the paper emphasises are direct registration and this data structure.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2107.06829).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
LiDAR + IMU → inertial propagation and scan motion correction → local map correspondences → iterated state update → pose and map update
```

In the review, separate the motion prediction produced by the IMU from the path by which LiDAR registration corrects that prediction. It builds a local map, but a global loop closure back-end is outside the scope of the original paper.

## 4. Experimental results and interpretation

The authors evaluate on 19 public sequences and on several LiDARs and platforms. The abstract reports up to 100 Hz odometry and mapping and estimation at 1000 deg/s rotation in particular experiments. These must not be read as figures guaranteed on every sensor and platform. [Source](https://arxiv.org/abs/2107.06829)

Accuracy and speed depend on the geometric structure of the environment and on the compute conditions. Long-range accumulated error without loops has to be distinguished from momentary estimation failure.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. How are per-point times, IMU times and extrinsics configured in the Vision60 logs?
2. Does the Z drift on step segments appear together with point cloud distortion, IMU anomalies, or a lack of geometric constraint?
3. What are the frame and the timestamp of the pose and point cloud passed on to a downstream back-end?

**Proposed experiment:** split the same log into flat, rotating and landing segments and record relative pose error, height error and processing latency. Without ground truth, do not treat drift figures as accuracy; keep map overlap and differences between repeated traverses as auxiliary indicators.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Explain whether `direct` in FAST-LIO2 means “no ICP and no correspondence search”, and compare with feature-based LIO.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

That is not what it means. FAST-LIO2 skips hand-crafted edge/plane feature extraction and registers raw points directly to the map. It finds map neighbours around each point, builds a local plane, and forms a point-to-plane residual. The difference is closer to **whether the points that will form observations are pre-selected as features**. A good answer separates direct registration, correspondence search and residual computation from each other.

</details>

### Q2. Math and reasoning

Given a point \(p*i^L\) transformed into the world frame as \(p_i^W=R*{WI}(R*{IL}p_i^L+t*{IL})+t\_{WI}\) and a plane \(n_i^\top x+d_i=0\), write the LiDAR residual and explain for which states the Jacobian can degenerate.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The residual is \(r*i=n_i^\top p_i^W+d_i\). For a small rotation perturbation \(\delta\theta\), the rotation Jacobian takes the form \(-n_i^\top R*{WI}[R_{IL}p_i^L+t_{IL}]\_\times\) depending on the sign convention, and the position Jacobian the form \(n_i^\top\). If all the normals are similar, translation perpendicular to the normal is weakly observed, and rotation also becomes weak if the point distribution and normals are not diverse enough. The key is to connect small eigenvalues of the Jacobian or information matrix to geometric degeneracy.

</details>

### Q3. Systems and debugging

Z jumps right after landing on a stair. Which logs would you check, and in what order, before tuning the filter?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Check per-point timestamps and the LiDAR–IMU time base, IMU clipping and drops, extrinsics, the point cloud before and after deskew, residuals and the selected plane normals, and covariance, in that order. Replay the same raw log to get reproducibility and separate before and after the landing. Simply reducing the process noise may smooth the output while hiding a bias or a timing error and increasing latency.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/hku-mars/FAST_LIO)

You have to support a new LiDAR's `PointCloud2` in the repository. When the per-point time field has different units from the existing sensor, which processing paths and settings would you trace, and what regression tests would you write?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Start from `lidar_type`, `timestamp_unit` and the topic settings in `config`, and trace through the per-sensor callback and point-time conversion in `src/preprocess.cpp`, measurement synchronisation in `src/laserMapping.cpp`, and undistortion in `src/IMU_Processing.hpp`. Normalise the internal time unit to one, and check that the scan start and end times and the per-point relative times are monotonic. Replay a stationary bag, a constant-angular-rate bag and a bag with a known timestamp offset, and compare plane thickness and pose after deskew. A test that only checks whether it compiles will not catch a time-unit error.

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

**Read next:** [FAST-LIO review]({{ '/study/slam/state-estimation/fast-lio/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
